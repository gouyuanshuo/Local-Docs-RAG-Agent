from __future__ import annotations

import contextlib
import importlib
import inspect
import multiprocessing
import os
import pathlib
import select
import sys
import threading
import traceback
from typing import Any

import portalocker
import pytest
import qdrant_client

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag
from local_docs_rag_agent.core import exceptions, models
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import (
    index_ownership,
    ingest,
    manifest,
    pipeline,
    qdrant_store,
    store_factory,
)


class _FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    @property
    def status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="live")


class _LifecycleStore:
    supports_incremental_updates = False
    requires_live_embeddings = False

    def __init__(self, manifest_path: pathlib.Path) -> None:
        self.manifest_path = manifest_path
        self.save_calls = 0
        self.search_calls = 0
        self.dirty_at_save: list[bool] = []
        self.removed_source_paths: list[str] = []
        self.replaced_source_paths: list[str] = []
        self.writer_entered: threading.Event | None = None
        self.release_writer: threading.Event | None = None
        self.reader_entered: threading.Event | None = None

    def collection_exists(self) -> bool:
        return True

    def exists(self) -> bool:
        return self.collection_exists()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        del chunks
        self.save_calls += 1
        stored = manifest.IngestManifest.load(self.manifest_path)
        self.dirty_at_save.append(stored.repair_required)
        self.removed_source_paths = removed_source_paths or []
        self.replaced_source_paths = replaced_source_paths or []
        if self.writer_entered is not None:
            self.writer_entered.set()
        if self.release_writer is not None:
            assert self.release_writer.wait(timeout=5.0)

    def search(self, query: str, top_k: int) -> list[models.RetrievalHit]:
        del query, top_k
        self.search_calls += 1
        if self.reader_entered is not None:
            self.reader_entered.set()
        return []

    @property
    def embedding_status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="live")


class _StatefulLifecycleStore(_LifecycleStore):
    def __init__(self, manifest_path: pathlib.Path) -> None:
        super().__init__(manifest_path)
        self.stored_source_paths: set[str] = set()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        super().save(
            chunks,
            removed_source_paths=removed_source_paths,
            replaced_source_paths=replaced_source_paths,
        )
        self.stored_source_paths.difference_update(removed_source_paths or [])
        self.stored_source_paths.difference_update(replaced_source_paths or [])
        self.stored_source_paths.update(chunk.source_path for chunk in chunks)


class _OwnershipStore:
    supports_incremental_updates = True
    requires_live_embeddings = True

    def __init__(
        self,
        exists: bool,
        *,
        save_error: BaseException | None = None,
        delete_error: BaseException | None = None,
        create_before_error: bool = True,
    ) -> None:
        self._exists = exists
        self.delete_calls = 0
        self.save_calls = 0
        self.save_error = save_error
        self.delete_error = delete_error
        self.create_before_error = create_before_error

    def collection_exists(self) -> bool:
        return self._exists

    def exists(self) -> bool:
        return self.collection_exists()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        del chunks, removed_source_paths, replaced_source_paths
        self.save_calls += 1
        if self.create_before_error:
            self._exists = True
        if self.save_error is not None:
            raise self.save_error
        self._exists = True

    def delete_collection(self) -> None:
        self.delete_calls += 1
        if self.delete_error is not None:
            raise self.delete_error
        self._exists = False

    @property
    def embedding_status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="live")


class _SharedOwnershipStore:
    supports_incremental_updates = True
    requires_live_embeddings = True

    def __init__(self, exists: Any) -> None:
        self._exists_value = exists

    def collection_exists(self) -> bool:
        with self._exists_value.get_lock():
            return bool(self._exists_value.value)

    def exists(self) -> bool:
        return self.collection_exists()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        del chunks, removed_source_paths, replaced_source_paths
        with self._exists_value.get_lock():
            self._exists_value.value = 1

    def delete_collection(self) -> None:
        with self._exists_value.get_lock():
            self._exists_value.value = 0

    @property
    def embedding_status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="live")


def _local_config(tmp_path: pathlib.Path) -> app_config.AppConfig:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    return app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
        embedding_api_key=None,
        vector_backend="local",
    )


def _qdrant_config(tmp_path: pathlib.Path) -> app_config.AppConfig:
    return _local_config(tmp_path).with_overrides(
        embedding_api_key="fake-key",
        vector_backend="qdrant",
        qdrant_url="https://qdrant.example",
        qdrant_collection="lifecycle-test",
    )


def _install_store(
    monkeypatch: pytest.MonkeyPatch,
    store: _LifecycleStore,
    *,
    qdrant: bool = False,
) -> None:
    store.supports_incremental_updates = qdrant
    store.requires_live_embeddings = qdrant
    if qdrant:
        monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _LifecycleStore)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )


def _guard_process(
    index_path: str,
    manifest_path: str,
    entered: Any,
    release: Any,
    ready: Any | None = None,
    begin: Any | None = None,
) -> None:
    config = app_config.AppConfig.from_env().with_overrides(
        index_path=pathlib.Path(index_path),
        ingest_manifest_path=pathlib.Path(manifest_path),
        embedding_api_key=None,
        vector_backend="local",
    )
    if ready is not None:
        ready.set()
    if begin is not None:
        begin.wait(timeout=5.0)
    with rag.index_guard(config):
        entered.set()
        release.wait(timeout=5.0)


def _initialize_contender(
    config: app_config.AppConfig,
    begin: Any,
    results: Any,
) -> None:
    assert begin.wait(timeout=5.0)
    try:
        rag.initialize_owned_qdrant_index(config)
    except exceptions.ConfigurationError:
        results.put("rejected")
    else:
        results.put("initialized")


def _try_file_lock_process(lock_path: str, results: Any) -> None:
    lock = portalocker.Lock(
        lock_path,
        mode="a",
        timeout=0.0,
        fail_when_locked=True,
    )
    try:
        lock.acquire()
    except portalocker.exceptions.LockException:
        results.put("blocked")
    else:
        results.put("acquired")
        lock.release()


def _enter_guard_from_thread(
    config: app_config.AppConfig,
    entered: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        with rag.index_guard(config):
            entered.set()
    except BaseException as exc:  # pragma: no cover - assertion reports it
        errors.append(exc)


def _acquire_mutex_from_thread(
    mutex: Any,
    acquired: threading.Event,
) -> None:
    if mutex.acquire(timeout=0.2):
        acquired.set()
        mutex.release()


def test_index_guard_is_public_reentrant_and_excludes_second_thread(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def enter_from_second_thread() -> None:
        try:
            with rag.index_guard(config):
                second_entered.set()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    with rag.index_guard(config), rag.index_guard(config):
        worker = threading.Thread(target=enter_from_second_thread)
        worker.start()
        assert second_entered.wait(timeout=0.1) is False
    assert second_entered.wait(timeout=5.0) is True
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert errors == []


def test_index_guard_serializes_spawned_processes(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    second_entered = context.Event()
    release_second = context.Event()
    first = context.Process(
        target=_guard_process,
        args=(
            str(config.index_path),
            str(config.ingest_manifest_path),
            first_entered,
            release_first,
        ),
    )
    second = context.Process(
        target=_guard_process,
        args=(
            str(config.index_path),
            str(config.ingest_manifest_path),
            second_entered,
            release_second,
        ),
    )
    first.start()
    try:
        assert first_entered.wait(timeout=5.0) is True
        second.start()
        assert second_entered.wait(timeout=0.1) is False
        release_first.set()
        assert second_entered.wait(timeout=5.0) is True
        release_second.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        release_first.set()
        release_second.set()
        if first.is_alive():
            first.terminate()
        if second.pid is not None and second.is_alive():
            second.terminate()
        first.join(timeout=5.0)
        if second.pid is not None:
            second.join(timeout=5.0)


@pytest.mark.parametrize("shared_resource", ["manifest", "store"])
def test_index_guard_serializes_partial_overlap_between_threads(
    tmp_path: pathlib.Path,
    shared_resource: str,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _local_config(first_root)
    second = _local_config(second_root)
    if shared_resource == "manifest":
        second = second.with_overrides(
            ingest_manifest_path=first.ingest_manifest_path
        )
    else:
        second = second.with_overrides(index_path=first.index_path)
    holder_entered = threading.Event()
    release_holder = threading.Event()
    contender_entered = threading.Event()
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with rag.index_guard(first):
                holder_entered.set()
                assert release_holder.wait(timeout=5.0)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    def contend() -> None:
        try:
            with rag.index_guard(second):
                contender_entered.set()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    holder = threading.Thread(target=hold)
    contender = threading.Thread(target=contend)
    holder.start()
    try:
        assert holder_entered.wait(timeout=5.0) is True
        contender.start()
        assert contender_entered.wait(timeout=0.1) is False
        release_holder.set()
        assert contender_entered.wait(timeout=5.0) is True
    finally:
        release_holder.set()
        holder.join(timeout=5.0)
        contender.join(timeout=5.0)
    assert not holder.is_alive()
    assert not contender.is_alive()
    assert errors == []


@pytest.mark.parametrize("shared_resource", ["manifest", "store"])
def test_index_guard_serializes_partial_overlap_between_spawned_processes(
    tmp_path: pathlib.Path,
    shared_resource: str,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _local_config(first_root)
    second = _local_config(second_root)
    if shared_resource == "manifest":
        second = second.with_overrides(
            ingest_manifest_path=first.ingest_manifest_path
        )
    else:
        second = second.with_overrides(index_path=first.index_path)
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    second_entered = context.Event()
    release_second = context.Event()
    second_ready = context.Event()
    begin_second = context.Event()
    holder = context.Process(
        target=_guard_process,
        args=(
            str(first.index_path),
            str(first.ingest_manifest_path),
            first_entered,
            release_first,
        ),
    )
    contender = context.Process(
        target=_guard_process,
        args=(
            str(second.index_path),
            str(second.ingest_manifest_path),
            second_entered,
            release_second,
            second_ready,
            begin_second,
        ),
    )
    holder.start()
    try:
        assert first_entered.wait(timeout=5.0) is True
        contender.start()
        assert second_ready.wait(timeout=5.0) is True
        begin_second.set()
        assert second_entered.wait(timeout=0.3) is False
        release_first.set()
        assert second_entered.wait(timeout=5.0) is True
        release_second.set()
        holder.join(timeout=5.0)
        contender.join(timeout=5.0)
        assert holder.exitcode == 0
        assert contender.exitcode == 0
    finally:
        release_first.set()
        release_second.set()
        begin_second.set()
        if holder.is_alive():
            holder.terminate()
        if contender.pid is not None and contender.is_alive():
            contender.terminate()
        holder.join(timeout=5.0)
        if contender.pid is not None:
            contender.join(timeout=5.0)


def test_local_guard_does_not_depend_on_process_temp_directory(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    context = multiprocessing.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    second_entered = context.Event()
    release_second = context.Event()
    second_ready = context.Event()
    begin_second = context.Event()
    first_temp = tmp_path / "first-temp"
    second_temp = tmp_path / "second-temp"
    first_temp.mkdir()
    second_temp.mkdir()

    monkeypatch.setenv("TMPDIR", str(first_temp))
    first = context.Process(
        target=_guard_process,
        args=(
            str(config.index_path),
            str(config.ingest_manifest_path),
            first_entered,
            release_first,
        ),
    )
    first.start()
    try:
        assert first_entered.wait(timeout=5.0) is True
        monkeypatch.setenv("TMPDIR", str(second_temp))
        second = context.Process(
            target=_guard_process,
            args=(
                str(config.index_path),
                str(config.ingest_manifest_path),
                second_entered,
                release_second,
                second_ready,
                begin_second,
            ),
        )
        second.start()
        assert second_ready.wait(timeout=5.0) is True
        begin_second.set()
        assert second_entered.wait(timeout=0.5) is False
        release_first.set()
        assert second_entered.wait(timeout=5.0) is True
        release_second.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        release_first.set()
        release_second.set()
        begin_second.set()
        if first.is_alive():
            first.terminate()
        if "second" in locals() and second.is_alive():
            second.terminate()
        first.join(timeout=5.0)
        if "second" in locals():
            second.join(timeout=5.0)


def test_qdrant_lock_artifacts_exclude_endpoint_credentials(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    remote_lock_root = tmp_path / "remote-locks"
    monkeypatch.setattr(index_lock, "_REMOTE_LOCK_ROOT", remote_lock_root)
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_url=(
            "https://lock-user:lock-password@qdrant.example/private-path"
            "?token=lock-query#lock-fragment"
        )
    )

    with rag.index_guard(config):
        pass

    artifacts = [
        *remote_lock_root.glob("*.lock"),
        *config.ingest_manifest_path.parent.glob("*.lock"),
    ]
    assert len(artifacts) == 2
    rendered = "\n".join(str(path) for path in artifacts)
    for secret in (
        "lock-user",
        "lock-password",
        "private-path",
        "lock-query",
        "lock-fragment",
    ):
        assert secret not in rendered


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_reacquires_os_lock_before_entering(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    read_fd, write_fd = os.pipe()
    child_pid: int | None = None
    try:
        with rag.index_guard(config):
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                try:
                    with rag.index_guard(config):
                        os.write(write_fd, b"entered")
                finally:
                    os.close(write_fd)
                os._exit(0)
            os.close(write_fd)
            write_fd = -1
            readable, _, _ = select.select([read_fd], [], [], 0.3)
            assert readable == []

        readable, _, _ = select.select([read_fd], [], [], 5.0)
        assert readable == [read_fd]
        assert os.read(read_fd, 7) == b"entered"
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
        child_pid = None
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, 9)
            os.waitpid(child_pid, 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_unwind_does_not_release_parent_lock(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    guard = rag.index_guard(config)
    guard.__enter__()
    guard_active = True
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        guard.__exit__(None, None, None)
        os.write(write_fd, b"unwound")
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    context = multiprocessing.get_context("fork")
    contender_entered = context.Event()
    release_contender = context.Event()
    contender: Any = None
    try:
        readable, _, _ = select.select([read_fd], [], [], 5.0)
        assert readable == [read_fd]
        assert os.read(read_fd, 7) == b"unwound"
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
        child_pid = -1

        contender = context.Process(
            target=_guard_process,
            args=(
                str(config.index_path),
                str(config.ingest_manifest_path),
                contender_entered,
                release_contender,
            ),
        )
        contender.start()
        assert contender_entered.wait(timeout=0.3) is False
        guard.__exit__(None, None, None)
        guard_active = False
        assert contender_entered.wait(timeout=5.0) is True
        release_contender.set()
        contender.join(timeout=5.0)
        assert contender.exitcode == 0
    finally:
        if guard_active:
            guard.__exit__(None, None, None)
        release_contender.set()
        os.close(read_fd)
        if child_pid > 0:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, 9)
            os.waitpid(child_pid, 0)
        if contender is not None and contender.is_alive():
            contender.terminate()
            contender.join(timeout=5.0)


def test_index_guard_normalizes_file_open_failure_without_release(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    released = False

    class _FailingFileLock:
        def acquire(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise PermissionError("synthetic denied path")

        def release(self) -> None:
            nonlocal released
            released = True
            raise AssertionError("unacquired lock must not be released")

    monkeypatch.setattr(
        index_lock.portalocker,
        "Lock",
        lambda *args, **kwargs: _FailingFileLock(),
    )

    with (
        pytest.raises(
            exceptions.ConfigurationError,
            match="index lifecycle lock",
        ),
        rag.index_guard(config),
    ):
        pytest.fail("unopenable lock unexpectedly acquired")
    assert released is False


def test_index_guard_releases_first_resource_when_second_acquisition_fails(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    locks: list[Any] = []

    class _OrderedFileLock:
        def __init__(self) -> None:
            self.released = False
            self.position = len(locks)
            self.fh = None
            locks.append(self)

        def acquire(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            if self.position == 1:
                raise PermissionError("synthetic second-resource failure")

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(
        index_lock.portalocker,
        "Lock",
        lambda *args, **kwargs: _OrderedFileLock(),
    )

    with (
        pytest.raises(exceptions.ConfigurationError),
        rag.index_guard(config),
    ):
        pytest.fail("partially acquired guard unexpectedly entered")

    assert len(locks) == 2
    assert locks[0].released is True
    assert locks[1].released is False


def test_index_guard_cleans_interrupted_file_lock_publication(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    interruption = KeyboardInterrupt("post-acquire registry interruption")
    created_locks: list[Any] = []
    original_lock = index_lock.portalocker.Lock

    class _InterruptingRegistry(dict[int, Any]):
        armed = True

        def __setitem__(self, key: int, value: Any) -> None:
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise interruption

    registry = _InterruptingRegistry()

    def tracked_lock(*args: Any, **kwargs: Any) -> Any:
        lock = original_lock(*args, **kwargs)
        created_locks.append(lock)
        return lock

    monkeypatch.setattr(index_lock, "_ACTIVE_FILE_LOCKS", registry)
    monkeypatch.setattr(index_lock.portalocker, "Lock", tracked_lock)

    with (
        pytest.raises(KeyboardInterrupt) as exc_info,
        rag.index_guard(config),
    ):
        pytest.fail("interrupted guard unexpectedly entered")

    assert exc_info.value is interruption
    assert registry == {}
    assert created_locks[0].fh is None
    with rag.index_guard(config):
        pass
    assert registry == {}


def test_index_guard_cleans_interrupted_thread_lock_publication(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resources, _ = index_lock._lock_resources(config)
    interruption = KeyboardInterrupt("post-thread-acquire interruption")

    class _InterruptAfterAcquire:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.armed = True

        def acquire(self, *, timeout: float) -> bool:
            acquired = self.lock.acquire(timeout=timeout)
            if acquired and self.armed:
                self.armed = False
                raise interruption
            return acquired

        def release(self) -> None:
            self.lock.release()

    interrupting_lock = _InterruptAfterAcquire()
    state = index_lock._LockState(thread_lock=interrupting_lock)
    index_lock._LOCK_STATES[resources[1].key] = state

    with (
        pytest.raises(KeyboardInterrupt) as exc_info,
        rag.index_guard(config),
    ):
        pytest.fail("interrupted guard unexpectedly entered")

    assert exc_info.value is interruption
    assert interrupting_lock.lock.acquire(blocking=False) is True
    interrupting_lock.lock.release()
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    with rag.index_guard(config):
        pass


@pytest.mark.parametrize("phase", ["before_increment", "after_increment"])
def test_nested_depth_interruption_preserves_outer_lock(
    tmp_path: pathlib.Path,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    interruption = KeyboardInterrupt("nested acquisition interrupted")
    source_lines, first_line = inspect.getsourcelines(
        index_lock._acquire_resource
    )
    before_markers = (
        "if not frame.state.thread_lock.acquire",
        "depths[key] = depths[key] + 1",
    )
    increment_offset = next(
        offset
        for offset, source_line in enumerate(source_lines)
        if any(marker in source_line for marker in before_markers)
    )
    target_offset = increment_offset
    if phase == "after_increment":
        target_offset = next(
            offset
            for offset, source_line in enumerate(source_lines)
            if offset > increment_offset and source_line.strip() == "return"
        )
    target_line = first_line + target_offset

    assert target_line == next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if offset == target_offset
    )

    def interrupt_nested(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is index_lock._acquire_resource.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            resource_frame = frame.f_locals["frame"]
            depths = getattr(index_lock._THREAD_STATE, "depths", {})
            if depths.get(resource_frame.resource.key, 0) > 0:
                interrupted_keys.append(resource_frame.resource.key)
                sys.settrace(None)
                raise interruption
        return interrupt_nested

    caught: BaseException | None = None
    outer_exit_error: BaseException | None = None
    outer_continued = False
    interrupted_keys: list[str] = []
    contender_acquired = threading.Event()
    contender_errors: list[BaseException] = []

    def contend_for_interrupted_resource() -> None:
        try:
            state = index_lock._LOCK_STATES[interrupted_keys[0]]
            if state.thread_lock.acquire(timeout=0.1):
                contender_acquired.set()
                state.thread_lock.release()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            contender_errors.append(exc)

    try:
        with rag.index_guard(config):
            sys.settrace(interrupt_nested)
            try:
                with rag.index_guard(config):
                    pytest.fail("interrupted nested guard unexpectedly entered")
            except KeyboardInterrupt as exc:
                caught = exc
            finally:
                sys.settrace(None)
            assert set(index_lock._thread_depths().values()) == {1}
            contender = threading.Thread(
                target=contend_for_interrupted_resource
            )
            contender.start()
            assert contender_acquired.wait(timeout=0.2) is False
            contender.join(timeout=5.0)
            assert not contender.is_alive()
            outer_continued = True
    except BaseException as exc:
        outer_exit_error = exc
    finally:
        sys.settrace(None)

    assert caught is interruption
    assert outer_continued is True
    assert contender_errors == []
    assert outer_exit_error is None
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    with rag.index_guard(config):
        pass


@pytest.mark.parametrize("phase", ["before_restore", "after_restore"])
def test_depth_release_interruption_restores_exact_state(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    shared = _local_config(shared_root).with_overrides(
        index_path=config.index_path
    )
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resources, _ = index_lock._lock_resources(config)
    target = next(
        resource.key
        for resource in resources
        if "local-store" in resource.path.name
    )
    interruption = KeyboardInterrupt(f"{phase} logical depth restoration")

    class _InterruptingDepths(dict[str, int]):
        armed = True

        def pop(self, key: str, *args: Any) -> int:
            if key != target or not self.armed:
                return super().pop(key, *args)
            self.armed = False
            if phase == "after_restore":
                super().pop(key, *args)
            raise interruption

    monkeypatch.setattr(index_lock, "LOCK_TIMEOUT_SECONDS", 0.2)
    with pytest.raises(KeyboardInterrupt) as exc_info, rag.index_guard(config):
        current = dict(index_lock._thread_depths())
        index_lock._THREAD_STATE.depths = _InterruptingDepths(current)

    assert exc_info.value is interruption
    assert index_lock._thread_depths().get(target, 0) == 0
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    assert index_lock._LOCK_STATES == {}
    entered = threading.Event()
    release_contender = threading.Event()
    errors: list[BaseException] = []

    def hold_shared_target() -> None:
        try:
            with rag.index_guard(shared):
                entered.set()
                release_contender.wait(timeout=5.0)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    contender = threading.Thread(target=hold_shared_target)
    contender.start()
    assert entered.wait(timeout=2.0) is True
    with (
        pytest.raises(exceptions.ConfigurationError) as timeout_info,
        rag.index_guard(config),
    ):
        pytest.fail("stale logical depth bypassed the physical lock")
    assert "timed out" in str(timeout_info.value).lower()
    release_contender.set()
    contender.join(timeout=5.0)
    assert not contender.is_alive()
    assert errors == []


def test_unresolved_depth_restore_retains_physical_ownership_until_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    shared = _local_config(shared_root).with_overrides(
        index_path=config.index_path
    )
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resources, _ = index_lock._lock_resources(config)
    target = next(
        resource
        for resource in resources
        if "local-store" in resource.path.name
    )
    restore_error = KeyboardInterrupt("logical depth restore interrupted")
    body_error = SystemExit("active lifecycle interruption")

    class _InterruptingDepths(dict[str, int]):
        armed = True

        def pop(self, key: str, *args: Any) -> int:
            if key == target.key and self.armed:
                self.armed = False
                raise restore_error
            return super().pop(key, *args)

    original_release = index_lock._release_resource
    first_cleanup_paused = threading.Event()
    allow_outer_retry = threading.Event()
    paused = False

    def pause_after_unresolved_release(frame: Any) -> None:
        nonlocal paused
        try:
            original_release(frame)
        finally:
            if frame.resource.key == target.key and not paused:
                paused = True
                first_cleanup_paused.set()
                allow_outer_retry.wait(timeout=5.0)

    monkeypatch.setattr(
        index_lock,
        "_release_resource",
        pause_after_unresolved_release,
    )
    escaped: list[BaseException] = []

    def interrupt_guard_body() -> None:
        try:
            with rag.index_guard(config):
                current = dict(index_lock._thread_depths())
                index_lock._THREAD_STATE.depths = _InterruptingDepths(current)
                raise body_error
        except BaseException as exc:
            escaped.append(exc)

    owner = threading.Thread(target=interrupt_guard_body)
    owner.start()
    contender_entered = threading.Event()
    contender_errors: list[BaseException] = []
    contender = threading.Thread(
        target=_enter_guard_from_thread,
        args=(shared, contender_entered, contender_errors),
    )
    try:
        assert first_cleanup_paused.wait(timeout=2.0) is True
        state = index_lock._LOCK_STATES[target.key]
        assert state.retainers
        contender.start()
        assert contender_entered.wait(timeout=0.3) is False
    finally:
        allow_outer_retry.set()
        owner.join(timeout=5.0)
        if contender.ident is not None:
            contender.join(timeout=5.0)

    assert not owner.is_alive()
    assert not contender.is_alive()
    assert escaped == [body_error]
    assert contender_entered.is_set()
    assert contender_errors == []
    assert index_lock._LOCK_STATES == {}


@pytest.mark.parametrize("phase", ["before_release", "after_release"])
def test_file_release_interruption_frees_os_lock(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    interruption = KeyboardInterrupt(f"{phase} file lock release")
    original_lock = index_lock.portalocker.Lock
    armed = True
    interrupted_paths: list[str] = []

    class _InterruptingFileLock:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = original_lock(*args, **kwargs)
            self.path = str(args[0])

        @property
        def fh(self) -> Any:
            return self._inner.fh

        @fh.setter
        def fh(self, value: Any) -> None:
            self._inner.fh = value

        def acquire(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner.acquire(*args, **kwargs)

        def release(self) -> None:
            nonlocal armed
            if armed:
                armed = False
                interrupted_paths.append(self.path)
                if phase == "after_release":
                    self._inner.release()
                raise interruption
            self._inner.release()

    monkeypatch.setattr(index_lock.portalocker, "Lock", _InterruptingFileLock)

    with pytest.raises(KeyboardInterrupt) as exc_info, rag.index_guard(config):
        pass

    escaped = exc_info.value
    assert escaped is interruption
    assert escaped.__traceback__ is not None
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    contender = context.Process(
        target=_try_file_lock_process,
        args=(interrupted_paths[0], results),
    )
    contender.start()
    contender.join(timeout=5.0)
    assert contender.exitcode == 0
    assert results.get(timeout=1.0) == "acquired"


def test_unresolved_file_close_retains_state_until_retry(
    tmp_path: pathlib.Path,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resource = index_lock._lock_resources(config)[0][0]
    primary = RuntimeError("file release failed before unlock")
    fallback = KeyboardInterrupt("file close fallback interrupted")

    class _InterruptingHandle:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise fallback

    class _InterruptedFileLock:
        def __init__(self) -> None:
            self.fh: Any = _InterruptingHandle()
            self.release_calls = 0

        def release(self) -> None:
            self.release_calls += 1
            if self.release_calls <= 2:
                raise primary
            self.fh = None

    file_lock = _InterruptedFileLock()
    frame = index_lock._AcquiredResource(resource=resource)
    state = index_lock._LockState()
    frame.state = state
    frame.file_lock = file_lock
    frame.file_lock_acquired = True
    state.retainers.add(frame.retention_token)
    index_lock._LOCK_STATES[resource.key] = state
    index_lock._ACTIVE_FILE_LOCKS[id(file_lock)] = file_lock

    with pytest.raises(RuntimeError) as exc_info:
        index_lock._release_resource(frame)

    assert exc_info.value is primary
    assert frame.file_lock_acquired is True
    assert index_lock._LOCK_STATES.get(resource.key) is state
    assert frame.retention_token in state.retainers

    index_lock._release_resource(frame)

    assert frame.file_lock_acquired is False
    assert resource.key not in index_lock._LOCK_STATES
    assert id(file_lock) not in index_lock._ACTIVE_FILE_LOCKS


@pytest.mark.parametrize("phase", ["before_release", "after_release"])
def test_thread_release_interruption_cleans_state_and_unlocks(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resources, _ = index_lock._lock_resources(config)
    target = resources[-1]
    interruption = KeyboardInterrupt(f"{phase} thread lock release")

    class _InterruptingThreadLock:
        def __init__(self) -> None:
            self.inner = threading.RLock()
            self.armed = True

        def acquire(self, *, timeout: float) -> bool:
            return self.inner.acquire(timeout=timeout)

        def release(self) -> None:
            if self.armed:
                self.armed = False
                if phase == "after_release":
                    self.inner.release()
                raise interruption
            self.inner.release()

    lock = _InterruptingThreadLock()
    index_lock._LOCK_STATES[target.key] = index_lock._LockState(
        thread_lock=lock
    )
    monkeypatch.setattr(index_lock, "LOCK_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(KeyboardInterrupt) as exc_info, rag.index_guard(config):
        pass

    assert exc_info.value is interruption
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    assert index_lock._LOCK_STATES == {}
    old_lock_acquired = threading.Event()

    def acquire_old_lock() -> None:
        if lock.inner.acquire(timeout=0.2):
            old_lock_acquired.set()
            lock.inner.release()

    old_lock_contender = threading.Thread(target=acquire_old_lock)
    old_lock_contender.start()
    assert old_lock_acquired.wait(timeout=2.0) is True
    old_lock_contender.join(timeout=5.0)
    assert not old_lock_contender.is_alive()
    entered = threading.Event()
    errors: list[BaseException] = []
    contender = threading.Thread(
        target=_enter_guard_from_thread,
        args=(config, entered, errors),
    )
    contender.start()
    assert entered.wait(timeout=2.0) is True
    contender.join(timeout=5.0)
    assert not contender.is_alive()
    assert errors == []


def test_unresolved_thread_release_retains_state_until_outer_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    shared = _local_config(shared_root).with_overrides(
        index_path=config.index_path
    )
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    resources, _ = index_lock._lock_resources(config)
    target = next(
        resource
        for resource in resources
        if "local-store" in resource.path.name
    )
    primary = RuntimeError("thread release failed before unlock")
    fallback = KeyboardInterrupt("thread release fallback interrupted")
    body_error = SystemExit("active lifecycle interruption")

    class _TwiceInterruptedThreadLock:
        def __init__(self) -> None:
            self.inner = threading.RLock()
            self.release_calls = 0

        def acquire(self, *, timeout: float) -> bool:
            return self.inner.acquire(timeout=timeout)

        def release(self) -> None:
            self.release_calls += 1
            if self.release_calls == 1:
                raise primary
            if self.release_calls == 2:
                raise fallback
            self.inner.release()

    lock = _TwiceInterruptedThreadLock()
    state = index_lock._LockState(thread_lock=lock)
    index_lock._LOCK_STATES[target.key] = state
    original_release = index_lock._release_resource
    first_cleanup_paused = threading.Event()
    allow_outer_retry = threading.Event()
    paused = False

    def pause_after_unresolved_release(frame: Any) -> None:
        nonlocal paused
        try:
            original_release(frame)
        finally:
            if frame.resource.key == target.key and not paused:
                paused = True
                first_cleanup_paused.set()
                allow_outer_retry.wait(timeout=5.0)

    monkeypatch.setattr(
        index_lock,
        "_release_resource",
        pause_after_unresolved_release,
    )
    escaped: list[BaseException] = []

    def interrupt_guard_body() -> None:
        try:
            with rag.index_guard(config):
                raise body_error
        except BaseException as exc:
            escaped.append(exc)

    owner = threading.Thread(target=interrupt_guard_body)
    owner.start()
    contender_entered = threading.Event()
    contender_errors: list[BaseException] = []
    contender = threading.Thread(
        target=_enter_guard_from_thread,
        args=(shared, contender_entered, contender_errors),
    )
    try:
        assert first_cleanup_paused.wait(timeout=2.0) is True
        assert index_lock._LOCK_STATES.get(target.key) is state
        assert state.retainers
        contender.start()
        assert contender_entered.wait(timeout=0.3) is False
    finally:
        allow_outer_retry.set()
        owner.join(timeout=5.0)
        if contender.ident is not None:
            contender.join(timeout=5.0)

    assert not owner.is_alive()
    assert not contender.is_alive()
    assert escaped == [body_error]
    assert contender_entered.is_set()
    assert contender_errors == []
    assert index_lock._LOCK_STATES == {}


@pytest.mark.parametrize("phase", ["before_release", "after_release"])
def test_registry_release_interruption_is_retried(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    inner = index_lock._REGISTRY_GUARD
    interruption = KeyboardInterrupt(f"{phase} registry release")

    class _InterruptingRegistryGuard:
        armed = False
        releases_before_interrupt = 0

        def acquire(self) -> bool:
            return bool(inner.acquire())

        def release(self) -> None:
            if self.armed:
                if self.releases_before_interrupt:
                    self.releases_before_interrupt -= 1
                else:
                    self.armed = False
                    if phase == "after_release":
                        inner.release()
                    raise interruption
            inner.release()

        def __enter__(self) -> _InterruptingRegistryGuard:
            self.acquire()
            return self

        def __exit__(self, *args: object) -> None:
            del args
            self.release()

    guard = _InterruptingRegistryGuard()
    monkeypatch.setattr(index_lock, "_REGISTRY_GUARD", guard)

    with pytest.raises(KeyboardInterrupt) as exc_info, rag.index_guard(config):
        guard.armed = True
        guard.releases_before_interrupt = 1

    assert exc_info.value is interruption
    entered = threading.Event()
    errors: list[BaseException] = []
    contender = threading.Thread(
        target=_enter_guard_from_thread,
        args=(config, entered, errors),
        daemon=True,
    )
    contender.start()
    assert entered.wait(timeout=2.0) is True
    contender.join(timeout=5.0)
    assert not contender.is_alive()
    assert errors == []
    assert index_lock._LOCK_STATES == {}


@pytest.mark.parametrize(
    ("mutex_name", "section_name"),
    [
        ("_REGISTRY_GUARD", "_registry_section"),
        ("_ACTIVE_LOCKS_GUARD", "_active_locks_section"),
    ],
)
def test_internal_mutex_body_interruption_preserves_outer_recursion(
    mutex_name: str,
    section_name: str,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    mutex = getattr(index_lock, mutex_name)
    section = getattr(index_lock, section_name)
    interruption = KeyboardInterrupt(f"{mutex_name} protected body")
    contender_acquired = threading.Event()
    outer_release_error: BaseException | None = None

    mutex.acquire()
    try:
        with pytest.raises(KeyboardInterrupt) as exc_info, section():
            raise interruption
        assert exc_info.value is interruption
        contender = threading.Thread(
            target=_acquire_mutex_from_thread,
            args=(mutex, contender_acquired),
        )
        contender.start()
        assert contender_acquired.wait(timeout=0.3) is False
        contender.join(timeout=5.0)
        assert not contender.is_alive()
    finally:
        try:
            mutex.release()
        except BaseException as exc:
            outer_release_error = exc

    assert outer_release_error is None
    after_release = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(mutex, contender_acquired),
    )
    contender_acquired.clear()
    after_release.start()
    assert contender_acquired.wait(timeout=2.0) is True
    after_release.join(timeout=5.0)
    assert not after_release.is_alive()


@pytest.mark.parametrize(
    ("mutex_name", "section_name"),
    [
        ("_REGISTRY_GUARD", "_registry_section"),
        ("_ACTIVE_LOCKS_GUARD", "_active_locks_section"),
    ],
)
def test_internal_mutex_pending_release_is_drained_before_reentry(
    monkeypatch: pytest.MonkeyPatch,
    mutex_name: str,
    section_name: str,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    inner = getattr(index_lock, mutex_name)
    section = getattr(index_lock, section_name)
    primary = RuntimeError(f"{mutex_name} primary release failure")
    fallback = KeyboardInterrupt(f"{mutex_name} fallback interruption")

    class _TwiceInterruptedMutex:
        release_calls = 0

        def acquire(self, *args: Any, **kwargs: Any) -> Any:
            return inner.acquire(*args, **kwargs)

        def release(self) -> None:
            self.release_calls += 1
            if self.release_calls == 1:
                raise primary
            if self.release_calls == 2:
                raise fallback
            inner.release()

        def __enter__(self) -> _TwiceInterruptedMutex:
            self.acquire()
            return self

        def __exit__(self, *args: object) -> None:
            del args
            self.release()

    interrupted = _TwiceInterruptedMutex()
    monkeypatch.setattr(index_lock, mutex_name, interrupted)

    with pytest.raises(RuntimeError) as exc_info, section():
        pass
    assert exc_info.value is primary

    with section():
        pass

    contender_acquired = threading.Event()
    contender = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(inner, contender_acquired),
    )
    contender.start()
    assert contender_acquired.wait(timeout=2.0) is True
    contender.join(timeout=5.0)
    assert not contender.is_alive()


@pytest.mark.parametrize("marker", ["depths[key] = 1", "body_error = None"])
def test_internal_mutex_pre_yield_interruption_releases_acquisition(
    marker: str,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    mutex = threading.RLock()
    source_lines, first_line = inspect.getsourcelines(index_lock._mutex_section)
    depth_offset = next(
        offset
        for offset, source_line in enumerate(source_lines)
        if "depths[key] = 1" in source_line
    )
    target_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if marker in source_line
        and (marker != "body_error = None" or offset > depth_offset)
    )
    interruption = KeyboardInterrupt(f"pre-yield interruption at {marker}")

    def interrupt_pre_yield(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is index_lock._mutex_section.__wrapped__.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interruption
        return interrupt_pre_yield

    sys.settrace(interrupt_pre_yield)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as exc_info,
            index_lock._mutex_section(mutex),
        ):
            pytest.fail("interrupted mutex section unexpectedly entered")
    finally:
        sys.settrace(None)

    assert exc_info.value is interruption
    contender_acquired = threading.Event()
    contender = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(mutex, contender_acquired),
    )
    contender.start()
    try:
        assert contender_acquired.wait(timeout=2.0) is True
        contender.join(timeout=5.0)
        assert not contender.is_alive()
        assert index_lock._mutex_depths() == {}
        assert index_lock._mutex_obligations() == []
    finally:
        with contextlib.suppress(RuntimeError):
            mutex.release()


def test_internal_mutex_obligation_publication_is_transactional() -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    mutex = threading.RLock()
    source_lines, first_line = inspect.getsourcelines(index_lock._mutex_section)
    append_offset = next(
        offset
        for offset, source_line in enumerate(source_lines)
        if "append(obligation)" in source_line
    )
    target_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if offset > append_offset and "body_error" in source_line
    )
    interruption = KeyboardInterrupt("post-obligation publication interruption")

    def interrupt_publication(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is index_lock._mutex_section.__wrapped__.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interruption
        return interrupt_publication

    sys.settrace(interrupt_publication)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as exc_info,
            index_lock._mutex_section(mutex),
        ):
            pytest.fail("interrupted mutex section unexpectedly entered")
    finally:
        sys.settrace(None)

    assert exc_info.value is interruption
    assert index_lock._mutex_depths() == {}
    assert index_lock._mutex_obligations() == []
    with index_lock._mutex_section(mutex):
        pass


@pytest.mark.parametrize(
    "phase",
    ["cleanup_entry", "before_depth_restore", "after_depth_restore"],
)
def test_internal_mutex_cleanup_transition_retries_whole_transition(
    phase: str,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    mutex = threading.RLock()
    cleanup = index_lock._finish_mutex_cleanup
    source_lines, first_line = inspect.getsourcelines(cleanup)
    markers = {
        "cleanup_entry": ("release_pending =",),
        "before_depth_restore": ("_restore_mutex_depth(key, depth_before)",),
        "after_depth_restore": ('obligation.phase = "cleanup"',),
    }
    offsets = [
        offset
        for offset, source_line in enumerate(source_lines)
        if any(marker in source_line for marker in markers[phase])
    ]
    target_line = first_line + offsets[-1]
    interruption = KeyboardInterrupt(f"{phase} mutex cleanup interruption")

    def interrupt_cleanup(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is cleanup.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interruption
        return interrupt_cleanup

    sys.settrace(interrupt_cleanup)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as exc_info,
            index_lock._mutex_section(mutex),
        ):
            pass
    finally:
        sys.settrace(None)

    assert exc_info.value is interruption
    with index_lock._mutex_section(mutex):
        pass

    contender_acquired = threading.Event()
    contender = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(mutex, contender_acquired),
    )
    contender.start()
    try:
        assert contender_acquired.wait(timeout=2.0) is True
        contender.join(timeout=5.0)
        assert not contender.is_alive()
        assert index_lock._mutex_depths() == {}
        assert index_lock._mutex_obligations() == []
    finally:
        with contextlib.suppress(RuntimeError):
            mutex.release()
        index_lock._THREAD_STATE.mutex_depths = {}
        index_lock._THREAD_STATE.mutex_obligations = []


@pytest.mark.parametrize("origin", ["acquire", "body"])
def test_mutex_cleanup_interruption_preserves_active_primary(
    origin: str,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    inner = threading.RLock()
    primary = SystemExit(f"{origin} primary")
    cleanup_interruption = KeyboardInterrupt("cleanup transition interrupted")

    class _AcquireInterruptingMutex:
        armed = True

        def acquire(self) -> bool:
            acquired = inner.acquire()
            if origin == "acquire" and self.armed:
                self.armed = False
                raise primary
            return acquired

        def release(self) -> None:
            inner.release()

    mutex = _AcquireInterruptingMutex()
    cleanup = index_lock._finish_mutex_cleanup
    source_lines, first_line = inspect.getsourcelines(cleanup)
    target_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if "release_pending =" in source_line
    )

    def interrupt_cleanup(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is cleanup.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise cleanup_interruption
        return interrupt_cleanup

    sys.settrace(interrupt_cleanup)
    try:
        with (
            pytest.raises(SystemExit) as exc_info,
            index_lock._mutex_section(mutex),
        ):
            if origin == "body":
                raise primary
    finally:
        sys.settrace(None)

    assert exc_info.value is primary
    contender_acquired = threading.Event()
    contender = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(inner, contender_acquired),
    )
    contender.start()
    assert contender_acquired.wait(timeout=2.0) is True
    contender.join(timeout=5.0)
    assert not contender.is_alive()
    assert index_lock._mutex_depths() == {}
    assert index_lock._mutex_obligations() == []


def test_mutex_cleanup_compensation_has_no_interruptible_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    inner = threading.RLock()
    primary = RuntimeError("primary cleanup release failure")
    decision_interruption = KeyboardInterrupt("cleanup decision interrupted")

    class _CleanupFailingMutex:
        release_calls = 0

        def acquire(self) -> bool:
            return inner.acquire()

        def release(self) -> None:
            self.release_calls += 1
            if self.release_calls <= 2:
                raise primary
            inner.release()

    mutex = _CleanupFailingMutex()
    decision_calls = 0

    def interrupt_cleanup_decision(*args: Any) -> bool:
        nonlocal decision_calls
        del args
        decision_calls += 1
        raise decision_interruption

    monkeypatch.setattr(
        index_lock,
        "_mutex_cleanup_complete",
        interrupt_cleanup_decision,
        raising=False,
    )

    escaped: BaseException | None = None
    try:
        with index_lock._mutex_section(mutex):
            pass
    except BaseException as exc:
        escaped = exc

    assert escaped is primary
    assert decision_calls == 0
    contender_acquired = threading.Event()
    contender = threading.Thread(
        target=_acquire_mutex_from_thread,
        args=(inner, contender_acquired),
    )
    contender.start()
    assert contender_acquired.wait(timeout=2.0) is True
    contender.join(timeout=5.0)
    assert not contender.is_alive()
    assert index_lock._mutex_depths() == {}
    assert index_lock._mutex_obligations() == []


@pytest.mark.parametrize(
    "phase",
    ["before_state", "after_state", "after_token", "after_registry"],
)
def test_retention_interruption_is_transactional(
    tmp_path: pathlib.Path,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    source_lines, first_line = inspect.getsourcelines(
        index_lock._retain_lock_state
    )
    markers = {
        "before_state": ("frame.state = state",),
        "after_state": ("state.retainers.add",),
        "after_token": ("_LOCK_STATES[key] = state",),
        "after_registry": ("return",),
    }
    target_line = next(
        first_line + offset
        for offset, source_line in enumerate(source_lines)
        if any(marker in source_line for marker in markers[phase])
    )
    interruption = KeyboardInterrupt(f"{phase} state retention")

    def interrupt_retain(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is index_lock._retain_lock_state.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interruption
        return interrupt_retain

    sys.settrace(interrupt_retain)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as exc_info,
            rag.index_guard(config),
        ):
            pytest.fail("interrupted guard unexpectedly entered")
    finally:
        sys.settrace(None)

    assert exc_info.value is interruption
    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    with rag.index_guard(config):
        pass


@pytest.mark.parametrize("phase", ["before_frame", "after_frame"])
def test_frame_publication_interruption_is_transactional(
    tmp_path: pathlib.Path,
    phase: str,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    guard_body = index_lock.index_guard.__wrapped__
    source_lines, first_line = inspect.getsourcelines(guard_body)
    append_offset = next(
        offset
        for offset, source_line in enumerate(source_lines)
        if "acquired.append(frame)" in source_line
    )
    target_offset = append_offset
    if phase == "after_frame":
        target_offset = next(
            offset
            for offset, source_line in enumerate(source_lines)
            if offset > append_offset
            and "_retain_lock_state(frame)" in source_line
        )
    target_line = first_line + target_offset
    interruption = KeyboardInterrupt(f"{phase} cleanup frame publication")

    def interrupt_frame(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            frame.f_code is guard_body.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interruption
        return interrupt_frame

    sys.settrace(interrupt_frame)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as exc_info,
            rag.index_guard(config),
        ):
            pytest.fail("interrupted guard unexpectedly entered")
    finally:
        sys.settrace(None)

    assert exc_info.value is interruption
    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}
    with rag.index_guard(config):
        pass


def test_reciprocal_partial_overlap_nesting_fails_immediately(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _local_config(first_root)
    second = _local_config(second_root)
    first_inner = first.with_overrides(index_path=second.index_path)
    second_inner = second.with_overrides(index_path=first.index_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    entered: list[str] = []
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    monkeypatch.setattr(index_lock, "LOCK_TIMEOUT_SECONDS", 0.2)

    def nest(
        label: str,
        outer: app_config.AppConfig,
        inner: app_config.AppConfig,
    ) -> None:
        try:
            with rag.index_guard(outer):
                barrier.wait(timeout=5.0)
                with rag.index_guard(inner):
                    entered.append(label)
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(
        target=nest,
        args=("first", first, first_inner),
    )
    second_thread = threading.Thread(
        target=nest,
        args=("second", second, second_inner),
    )
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=5.0)
    second_thread.join(timeout=5.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert entered == []
    assert len(errors) == 2
    assert all(isinstance(exc, exceptions.ConfigurationError) for exc in errors)
    assert all("unsafe nested" in str(exc).lower() for exc in errors)
    assert all(
        isinstance(exc, exceptions.ConfigurationError) and exc.action_hint
        for exc in errors
    )
    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}


def test_monotonic_disjoint_nested_guards_remain_allowed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _local_config(tmp_path)
    inner_root = tmp_path / "inner"
    inner_root.mkdir()
    inner = _local_config(inner_root)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    outer_resources = (
        index_lock._LockResource("a", tmp_path / "a.lock"),
        index_lock._LockResource("b", tmp_path / "b.lock"),
    )
    inner_resources = (
        index_lock._LockResource("c", tmp_path / "c.lock"),
        index_lock._LockResource("d", tmp_path / "d.lock"),
    )

    def resources_for(
        config: app_config.AppConfig,
    ) -> tuple[tuple[Any, ...], pathlib.Path]:
        if config is outer:
            return outer_resources, config.ingest_manifest_path
        assert config is inner
        return inner_resources, config.ingest_manifest_path

    monkeypatch.setattr(index_lock, "_lock_resources", resources_for)

    with rag.index_guard(outer), rag.index_guard(inner):
        assert set(index_lock._thread_depths()) == {"a", "b", "c", "d"}

    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}


def test_nested_subset_of_held_resources_remains_allowed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _local_config(tmp_path)
    inner_root = tmp_path / "inner"
    inner_root.mkdir()
    inner = _local_config(inner_root)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    outer_resources = (
        index_lock._LockResource("a", tmp_path / "a.lock"),
        index_lock._LockResource("b", tmp_path / "b.lock"),
    )
    inner_resources = (outer_resources[1],)

    def resources_for(
        config: app_config.AppConfig,
    ) -> tuple[tuple[Any, ...], pathlib.Path]:
        if config is outer:
            return outer_resources, config.ingest_manifest_path
        assert config is inner
        return inner_resources, config.ingest_manifest_path

    monkeypatch.setattr(index_lock, "_lock_resources", resources_for)

    with rag.index_guard(outer), rag.index_guard(inner):
        assert index_lock._thread_depths() == {"a": 1, "b": 2}

    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}


def test_reverse_order_disjoint_nested_guard_fails_before_retention(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _local_config(tmp_path)
    inner_root = tmp_path / "inner"
    inner_root.mkdir()
    inner = _local_config(inner_root)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    outer_resources = (
        index_lock._LockResource("c", tmp_path / "c.lock"),
        index_lock._LockResource("d", tmp_path / "d.lock"),
    )
    inner_resources = (
        index_lock._LockResource("a", tmp_path / "a.lock"),
        index_lock._LockResource("b", tmp_path / "b.lock"),
    )

    def resources_for(
        config: app_config.AppConfig,
    ) -> tuple[tuple[Any, ...], pathlib.Path]:
        if config is outer:
            return outer_resources, config.ingest_manifest_path
        assert config is inner
        return inner_resources, config.ingest_manifest_path

    monkeypatch.setattr(index_lock, "_lock_resources", resources_for)

    with rag.index_guard(outer):
        retained = dict(index_lock._LOCK_STATES)
        with (
            pytest.raises(exceptions.ConfigurationError) as exc_info,
            rag.index_guard(inner),
        ):
            pytest.fail("reverse-order nested guard unexpectedly entered")
        assert "unsafe nested" in str(exc_info.value).lower()
        assert retained == index_lock._LOCK_STATES

    assert index_lock._LOCK_STATES == {}
    assert index_lock._ACTIVE_FILE_LOCKS == {}


def test_index_guard_attempts_every_release_after_one_release_fails(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    locks: list[Any] = []

    class _OrderedFileLock:
        def __init__(self) -> None:
            self.release_attempted = False
            self.position = len(locks)
            self.fh = None
            locks.append(self)

        def acquire(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def release(self) -> None:
            self.release_attempted = True
            if self.position == 1:
                raise PermissionError("synthetic release failure")

    monkeypatch.setattr(
        index_lock.portalocker,
        "Lock",
        lambda *args, **kwargs: _OrderedFileLock(),
    )

    with (
        pytest.raises(exceptions.ConfigurationError),
        rag.index_guard(config),
    ):
        pass

    assert len(locks) == 2
    assert all(lock.release_attempted for lock in locks)


def test_index_guard_timeout_is_actionable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_guard_process,
        args=(
            str(config.index_path),
            str(config.ingest_manifest_path),
            entered,
            release,
        ),
    )
    holder.start()
    try:
        assert entered.wait(timeout=5.0) is True
        index_lock = importlib.import_module(
            "local_docs_rag_agent.rag.index_lock"
        )
        monkeypatch.setattr(index_lock, "LOCK_TIMEOUT_SECONDS", 0.05)
        with (
            pytest.raises(
                exceptions.ConfigurationError,
                match="index lifecycle lock",
            ) as exc_info,
            rag.index_guard(config),
        ):
            pytest.fail("contended lock unexpectedly acquired")
        assert exc_info.value.action_hint is not None
        assert "retry" in exc_info.value.action_hint.lower()
    finally:
        release.set()
        holder.join(timeout=5.0)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5.0)


def test_contended_target_does_not_block_independent_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_a = tmp_path / "target-a"
    target_b = tmp_path / "target-b"
    target_a.mkdir()
    target_b.mkdir()
    config_a = _local_config(target_a)
    config_b = _local_config(target_b)
    context = multiprocessing.get_context("spawn")
    holder_entered = context.Event()
    release_holder = context.Event()
    holder = context.Process(
        target=_guard_process,
        args=(
            str(config_a.index_path),
            str(config_a.ingest_manifest_path),
            holder_entered,
            release_holder,
        ),
    )
    holder.start()
    thread_a_attempted_file_lock = threading.Event()
    target_b_entered = threading.Event()
    errors: list[BaseException] = []
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    original_acquire = index_lock.portalocker.Lock.acquire

    def observed_acquire(self: Any, *args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name == "contended-target-a":
            thread_a_attempted_file_lock.set()
        return original_acquire(self, *args, **kwargs)

    monkeypatch.setattr(index_lock, "LOCK_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(
        index_lock.portalocker.Lock,
        "acquire",
        observed_acquire,
    )

    def enter(config: app_config.AppConfig, entered: threading.Event) -> None:
        try:
            with rag.index_guard(config):
                entered.set()
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    thread_a = threading.Thread(
        target=enter,
        args=(config_a, threading.Event()),
        name="contended-target-a",
    )
    thread_b = threading.Thread(
        target=enter,
        args=(config_b, target_b_entered),
        name="independent-target-b",
    )
    try:
        assert holder_entered.wait(timeout=5.0) is True
        thread_a.start()
        assert thread_a_attempted_file_lock.wait(timeout=5.0) is True
        thread_b.start()
        assert target_b_entered.wait(timeout=0.3) is True
    finally:
        release_holder.set()
        holder.join(timeout=5.0)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5.0)
        thread_a.join(timeout=5.0)
        thread_b.join(timeout=5.0)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []


def test_qdrant_ownership_initializer_rejects_local_and_existing_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(exceptions.ConfigurationError, match="Qdrant"):
        rag.initialize_owned_qdrant_index(_local_config(tmp_path))

    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="initializer-creates"
    )
    store = _OwnershipStore(exists=True)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    with pytest.raises(exceptions.ConfigurationError, match="already exists"):
        rag.initialize_owned_qdrant_index(config)
    assert store.delete_calls == 0


def test_qdrant_ownership_initializer_sanitizes_client_startup_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_url = (
        "https://owner-user:owner-password@qdrant.example/owner-path"
        "?token=owner-query#owner-fragment"
    )
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_url=configured_url,
        qdrant_api_key="owner-api-key",
        qdrant_collection="safe-owned-startup",
        external_http_trust_env=False,
    )
    raw_marker = "raw-owned-client-startup-message"

    class FailingQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError(
                f"{raw_marker}; client kwargs leaked: {kwargs!r}"
            )

    monkeypatch.setattr(qdrant_client, "QdrantClient", FailingQdrantClient)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    with pytest.raises(exceptions.VectorStoreError) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    rendered = "\n".join(
        (
            str(exc_info.value),
            repr(exc_info.value),
            "".join(traceback.format_exception(exc_info.value)),
        )
    )
    assert exc_info.value.reason_code == "operation_failed"
    assert "client_init" in str(exc_info.value)
    assert "safe-owned-startup" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    for secret in (
        "owner-user",
        "owner-password",
        "owner-path",
        "owner-query",
        "owner-fragment",
        "owner-api-key",
        configured_url,
        raw_marker,
        "kwargs leaked",
    ):
        assert secret not in rendered


def test_qdrant_ownership_initializer_creates_collection_before_return(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path)
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(exists=False)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    ownership = rag.initialize_owned_qdrant_index(config)

    assert isinstance(ownership, rag.QdrantIndexOwnership)
    assert store.exists() is True
    assert store.save_calls == 1
    rag.delete_owned_qdrant_index(config, ownership)


def test_qdrant_ownership_initializer_cleans_partial_failed_creation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="initializer-partial-failure"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(
        exists=False,
        save_error=exceptions.VectorStoreError(
            "forced first-ingest failure",
            reason_code="operation_failed",
        ),
    )
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    with pytest.raises(exceptions.VectorStoreError, match="forced"):
        rag.initialize_owned_qdrant_index(config)

    assert store.exists() is False
    assert store.delete_calls == 1
    store.save_error = None
    ownership = rag.initialize_owned_qdrant_index(config)
    rag.delete_owned_qdrant_index(config, ownership)


def test_qdrant_initializer_clears_claim_if_token_construction_is_interrupted(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="token-interruption"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(exists=False)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )
    original_token_type = index_ownership.QdrantIndexOwnership
    interruption = KeyboardInterrupt("token construction interrupted")

    def interrupt_token(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise interruption

    monkeypatch.setattr(
        index_ownership,
        "QdrantIndexOwnership",
        interrupt_token,
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    assert exc_info.value is interruption
    assert store.save_calls == 0
    assert store.delete_calls == 0
    assert index_ownership._CLAIMS == {}

    monkeypatch.setattr(
        index_ownership,
        "QdrantIndexOwnership",
        original_token_type,
    )
    ownership = rag.initialize_owned_qdrant_index(config)
    rag.delete_owned_qdrant_index(config, ownership)


def test_qdrant_initializer_clears_partially_registered_claim_on_interrupt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="registration-interruption"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(exists=False)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )
    interruption = KeyboardInterrupt("claim registration interrupted")

    class _InterruptingClaims(dict[str, tuple[str, str]]):
        armed = True

        def __setitem__(self, key: str, value: tuple[str, str]) -> None:
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise interruption

    claims = _InterruptingClaims()
    monkeypatch.setattr(index_ownership, "_CLAIMS", claims)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    assert exc_info.value is interruption
    assert store.save_calls == 0
    assert store.delete_calls == 0
    assert claims == {}

    ownership = rag.initialize_owned_qdrant_index(config)
    rag.delete_owned_qdrant_index(config, ownership)


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt("stop ingest"), SystemExit("stop ingest")],
)
def test_qdrant_initializer_cleans_and_re_raises_interruption(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="interrupted-initializer"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(exists=False, save_error=interruption)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    with pytest.raises(type(interruption)) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    assert exc_info.value is interruption
    assert store.exists() is False
    assert store.delete_calls == 1
    assert rag.qdrant_orphaned_collection(interruption) is None

    store.save_error = None
    ownership = rag.initialize_owned_qdrant_index(config)
    rag.delete_owned_qdrant_index(config, ownership)


@pytest.mark.parametrize(
    ("interruption", "cleanup_error"),
    [
        (KeyboardInterrupt("stop ingest"), RuntimeError("cleanup failed")),
        (SystemExit("stop ingest"), SystemExit("cleanup interrupted")),
    ],
)
def test_qdrant_initializer_marks_orphan_when_interrupt_cleanup_fails(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
    cleanup_error: BaseException,
) -> None:
    collection = "exact-interrupted-collection"
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection=collection
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(
        exists=False,
        save_error=interruption,
        delete_error=cleanup_error,
    )
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    with pytest.raises(type(interruption)) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    assert exc_info.value is interruption
    assert rag.qdrant_orphaned_collection(interruption) == collection
    assert any(collection in note for note in interruption.__notes__)
    assert store.exists() is True
    assert store.delete_calls == 1

    store._exists = False
    store.save_error = None
    store.delete_error = None
    ownership = rag.initialize_owned_qdrant_index(config)
    rag.delete_owned_qdrant_index(config, ownership)


def test_qdrant_initializer_does_not_mark_or_delete_absent_collection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interruption = KeyboardInterrupt("before collection creation")
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="never-created"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _OwnershipStore(
        exists=False,
        save_error=interruption,
        create_before_error=False,
    )
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        rag.initialize_owned_qdrant_index(config)

    assert exc_info.value is interruption
    assert store.delete_calls == 0
    assert rag.qdrant_orphaned_collection(interruption) is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_same_host_qdrant_contenders_cannot_both_initialize_absent_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path)
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    context = multiprocessing.get_context("fork")
    exists = context.Value("b", 0)
    store = _SharedOwnershipStore(exists)
    monkeypatch.setattr(
        qdrant_store,
        "QdrantChunkStore",
        _SharedOwnershipStore,
    )
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )
    begin = context.Event()
    results = context.Queue()
    contenders = [
        context.Process(
            target=_initialize_contender,
            args=(config, begin, results),
        )
        for _ in range(2)
    ]
    for contender in contenders:
        contender.start()
    try:
        begin.set()
        outcomes = sorted(results.get(timeout=5.0) for _ in contenders)
        for contender in contenders:
            contender.join(timeout=5.0)
            assert contender.exitcode == 0
        assert outcomes == ["initialized", "rejected"]
    finally:
        begin.set()
        for contender in contenders:
            if contender.is_alive():
                contender.terminate()
            contender.join(timeout=5.0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_cannot_use_inherited_qdrant_ownership(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="fork-owned-parent-only"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    context = multiprocessing.get_context("fork")
    exists = context.Value("b", 0)
    store = _SharedOwnershipStore(exists)
    monkeypatch.setattr(
        qdrant_store,
        "QdrantChunkStore",
        _SharedOwnershipStore,
    )
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )
    ownership = rag.initialize_owned_qdrant_index(config)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            rag.delete_owned_qdrant_index(config, ownership)
        except exceptions.ConfigurationError:
            os.write(write_fd, b"rejected")
        else:
            os.write(write_fd, b"deleted!")
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 5.0)
        assert readable == [read_fd]
        assert os.read(read_fd, 8) == b"rejected"
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
        child_pid = -1
        assert store.collection_exists() is True
    finally:
        os.close(read_fd)
        if child_pid > 0:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, 9)
            os.waitpid(child_pid, 0)
    rag.delete_owned_qdrant_index(config, ownership)
    assert store.collection_exists() is False


def test_qdrant_owned_cleanup_fails_closed_on_target_mismatch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="comparison-run-owned-1234"
    )
    baseline_config = run_config.with_overrides(
        qdrant_collection="interactive-baseline"
    )
    stores = {
        run_config.qdrant_collection: _OwnershipStore(exists=False),
        baseline_config.qdrant_collection: _OwnershipStore(exists=True),
    }
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: stores[
            config.qdrant_collection
        ],
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    ownership = rag.initialize_owned_qdrant_index(run_config)

    with pytest.raises(exceptions.ConfigurationError, match="ownership"):
        rag.delete_owned_qdrant_index(baseline_config, ownership)
    assert stores[baseline_config.qdrant_collection].delete_calls == 0

    rag.delete_owned_qdrant_index(run_config, ownership)
    assert stores[run_config.qdrant_collection].delete_calls == 1
    assert stores[baseline_config.qdrant_collection].exists() is True
    with pytest.raises(exceptions.ConfigurationError, match="ownership"):
        rag.delete_owned_qdrant_index(run_config, ownership)


def test_qdrant_delete_failures_retain_ownership_for_successful_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _qdrant_config(tmp_path).with_overrides(
        qdrant_collection="comparison-run-owned-retry"
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")

    class _FakeQdrantClient:
        def __init__(self) -> None:
            self.exists = False
            self.delete_results: list[object] = [
                RuntimeError("forced delete exception"),
                False,
                True,
            ]

        def collection_exists(self, collection_name: str) -> bool:
            del collection_name
            return self.exists

        def create_collection(self, **kwargs: Any) -> bool:
            del kwargs
            self.exists = True
            return True

        def create_payload_index(self, **kwargs: Any) -> bool:
            del kwargs
            return True

        def upsert(self, **kwargs: Any) -> bool:
            del kwargs
            return True

        def delete(self, **kwargs: Any) -> bool:
            del kwargs
            return True

        def delete_collection(self, collection_name: str) -> bool:
            del collection_name
            result = self.delete_results.pop(0)
            if isinstance(result, Exception):
                raise result
            if result:
                self.exists = False
            return bool(result)

    client = _FakeQdrantClient()
    monkeypatch.setattr(
        qdrant_client,
        "QdrantClient",
        lambda **kwargs: client,
    )
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: _FakeEmbeddingProvider(),
    )

    ownership = rag.initialize_owned_qdrant_index(config)
    with pytest.raises(exceptions.VectorStoreError):
        rag.delete_owned_qdrant_index(config, ownership)
    with pytest.raises(exceptions.VectorStoreError):
        rag.delete_owned_qdrant_index(config, ownership)

    rag.delete_owned_qdrant_index(config, ownership)
    assert client.exists is False
    with pytest.raises(exceptions.ConfigurationError, match="ownership"):
        rag.delete_owned_qdrant_index(config, ownership)


@pytest.mark.parametrize("backend", ["local", "qdrant"])
def test_dirty_intent_precedes_store_save(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    config = (
        _local_config(tmp_path)
        if backend == "local"
        else _qdrant_config(tmp_path)
    )
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _LifecycleStore(config.ingest_manifest_path)
    _install_store(monkeypatch, store, qdrant=backend == "qdrant")

    ingest.ingest_documents(config)

    assert store.dirty_at_save == [True]
    assert (
        manifest.IngestManifest.load(
            config.ingest_manifest_path
        ).repair_required
        is False
    )


@pytest.mark.parametrize("transition", ["removed", "changed_to_empty"])
def test_qdrant_transition_publishes_dirty_intent_before_save(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    config = _qdrant_config(tmp_path)
    source = config.docs_dir / "sample.md"
    source.write_text("Evidence", encoding="utf-8")
    store = _LifecycleStore(config.ingest_manifest_path)
    _install_store(monkeypatch, store, qdrant=True)
    ingest.ingest_documents(config)
    store.dirty_at_save.clear()

    if transition == "removed":
        source.unlink()
    else:
        source.write_text("   \n", encoding="utf-8")
    ingest.ingest_documents(config)

    assert store.dirty_at_save == [True]
    if transition == "removed":
        assert store.removed_source_paths == [source.as_posix()]
    else:
        assert store.replaced_source_paths == [source.as_posix()]


def test_empty_local_ingest_publishes_dirty_intent_before_save(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    store = _LifecycleStore(config.ingest_manifest_path)
    _install_store(monkeypatch, store)

    ingest.ingest_documents(config)

    assert store.dirty_at_save == [True]


def test_dirty_publication_failure_prevents_store_mutation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _LifecycleStore(config.ingest_manifest_path)
    _install_store(monkeypatch, store)

    def fail_save(
        self: manifest.IngestManifest,
        path: pathlib.Path,
    ) -> None:
        del self, path
        raise OSError("forced dirty publication failure")

    monkeypatch.setattr(manifest.IngestManifest, "save", fail_save)

    with pytest.raises(OSError, match="forced dirty publication failure"):
        ingest.ingest_documents(config)
    assert store.save_calls == 0


def test_reader_repairs_crash_after_local_replacement(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    original_save = manifest.IngestManifest.save
    fail_clean_publication = True

    def save_then_crash(
        self: manifest.IngestManifest,
        path: pathlib.Path,
    ) -> None:
        if fail_clean_publication and not self.repair_required:
            raise RuntimeError("simulated crash before clean publication")
        original_save(self, path)

    monkeypatch.setattr(manifest.IngestManifest, "save", save_then_crash)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest.ingest_documents(config)
    assert (
        manifest.IngestManifest.load(
            config.ingest_manifest_path
        ).repair_required
        is True
    )

    fail_clean_publication = False
    ingest.ensure_index(config)

    stored = manifest.IngestManifest.load(config.ingest_manifest_path)
    assert stored.repair_required is False
    stored_chunks = store_factory.build_store(config).load()
    assert [chunk.text for chunk in stored_chunks] == ["Evidence"]


@pytest.mark.parametrize("transition", ["removed", "excluded"])
def test_qdrant_repair_deletes_dirty_only_source_that_left_scope(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    config = _qdrant_config(tmp_path)
    source = config.docs_dir / "new.md"
    source.write_text("New evidence", encoding="utf-8")
    store = _StatefulLifecycleStore(config.ingest_manifest_path)
    _install_store(monkeypatch, store, qdrant=True)
    original_save = manifest.IngestManifest.save
    fail_clean_publication = True

    def save_then_crash(
        self: manifest.IngestManifest,
        path: pathlib.Path,
    ) -> None:
        if fail_clean_publication and not self.repair_required:
            raise RuntimeError("simulated crash before clean publication")
        original_save(self, path)

    monkeypatch.setattr(manifest.IngestManifest, "save", save_then_crash)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest.ingest_documents(config)
    assert store.stored_source_paths == {source.as_posix()}

    fail_clean_publication = False
    if transition == "removed":
        source.unlink()
        recovery_config = config
    else:
        recovery_config = config.with_overrides(
            docs_exclude_patterns=["new.md"]
        )
    ingest.ensure_index(recovery_config)

    assert store.stored_source_paths == set()
    assert store.removed_source_paths == [source.as_posix()]


def test_writer_blocks_reader_readiness_and_query(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    (config.docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    store = _LifecycleStore(config.ingest_manifest_path)
    store.writer_entered = threading.Event()
    store.release_writer = threading.Event()
    store.reader_entered = threading.Event()
    _install_store(monkeypatch, store)
    errors: list[BaseException] = []

    def write() -> None:
        try:
            ingest.ingest_documents(config)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    def read() -> None:
        try:
            pipeline.retrieve(config, "evidence")
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    writer = threading.Thread(target=write)
    reader = threading.Thread(target=read)
    writer.start()
    assert store.writer_entered.wait(timeout=5.0) is True
    reader.start()
    assert store.reader_entered.wait(timeout=0.1) is False
    store.release_writer.set()
    assert store.reader_entered.wait(timeout=5.0) is True
    writer.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []


def test_two_ingests_serialize_until_clean_manifest_publication(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_a = _local_config(tmp_path).with_overrides(
        chunk_strategy="fixed", chunk_size=20, chunk_overlap=5
    )
    config_b = config_a.with_overrides(chunk_size=30)
    (config_a.docs_dir / "sample.md").write_text(
        "Evidence that spans more than one chunk.", encoding="utf-8"
    )
    store = _LifecycleStore(config_a.ingest_manifest_path)
    _install_store(monkeypatch, store)
    first_before_clean = threading.Event()
    release_first = threading.Event()
    second_writer_entered = threading.Event()
    original_save = manifest.IngestManifest.save
    errors: list[BaseException] = []

    def controlled_save(
        self: manifest.IngestManifest,
        path: pathlib.Path,
    ) -> None:
        if not self.repair_required:
            if threading.current_thread().name == "writer-a":
                first_before_clean.set()
                assert release_first.wait(timeout=5.0)
            elif threading.current_thread().name == "writer-b":
                second_writer_entered.set()
        original_save(self, path)

    monkeypatch.setattr(manifest.IngestManifest, "save", controlled_save)

    def run(config: app_config.AppConfig) -> None:
        try:
            ingest.ingest_documents(config)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    first = threading.Thread(target=run, args=(config_a,), name="writer-a")
    second = threading.Thread(target=run, args=(config_b,), name="writer-b")
    first.start()
    assert first_before_clean.wait(timeout=5.0) is True
    second.start()
    assert second_writer_entered.wait(timeout=0.1) is False
    release_first.set()
    assert second_writer_entered.wait(timeout=5.0) is True
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    stored = manifest.IngestManifest.load(config_a.ingest_manifest_path)
    assert stored.index_fingerprint == ingest.index_fingerprint(
        config_b, embedding_mode="live"
    )
