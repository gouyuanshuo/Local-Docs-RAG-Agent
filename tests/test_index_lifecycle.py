from __future__ import annotations

import importlib
import multiprocessing
import pathlib
import threading
from typing import Any

import pytest

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag
from local_docs_rag_agent.core import exceptions, models
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import (
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


class _OwnershipStore:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.delete_calls = 0

    def collection_exists(self) -> bool:
        return self.exists

    def delete_collection(self) -> None:
        self.delete_calls += 1
        self.exists = False


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


def test_index_guard_normalizes_file_open_failure_without_release(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _local_config(tmp_path)
    index_lock = importlib.import_module("local_docs_rag_agent.rag.index_lock")
    released = False

    class _FailingFileLock:
        def acquire(self) -> None:
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


def test_qdrant_ownership_claim_rejects_local_and_existing_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(exceptions.ConfigurationError, match="Qdrant"):
        rag.claim_qdrant_index_ownership(_local_config(tmp_path))

    config = _qdrant_config(tmp_path)
    store = _OwnershipStore(exists=True)
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", _OwnershipStore)
    monkeypatch.setattr(store_factory, "build_store", lambda config: store)
    with pytest.raises(exceptions.ConfigurationError, match="already exists"):
        rag.claim_qdrant_index_ownership(config)
    assert store.delete_calls == 0


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
        lambda config: stores[config.qdrant_collection],
    )

    ownership = rag.claim_qdrant_index_ownership(run_config)
    stores[run_config.qdrant_collection].exists = True

    with pytest.raises(exceptions.ConfigurationError, match="ownership"):
        rag.delete_owned_qdrant_index(baseline_config, ownership)
    assert stores[baseline_config.qdrant_collection].delete_calls == 0

    rag.delete_owned_qdrant_index(run_config, ownership)
    assert stores[run_config.qdrant_collection].delete_calls == 1
    assert stores[baseline_config.qdrant_collection].exists is True
    with pytest.raises(exceptions.ConfigurationError, match="ownership"):
        rag.delete_owned_qdrant_index(run_config, ownership)


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
