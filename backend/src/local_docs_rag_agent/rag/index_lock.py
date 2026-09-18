"""Single-host serialization for manifest and storage lifecycle resources.

The operating-system lock coordinates processes while an explicit reentrant
thread lock closes the gap left by advisory locks inside one process. Manifest
and local-store locks live beside their canonical files; remote locks use a
credential-free identity in the user's cache. Resources are globally ordered,
and only the owning thread may enter recursively. Lock files are stable
identity markers and may safely outlive a run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import portalocker

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.core import exceptions

LOCK_TIMEOUT_SECONDS = 10.0
_REMOTE_LOCK_ROOT = (
    pathlib.Path.home() / ".cache" / "local-docs-rag-agent" / "locks"
)
_PROCESS_ID = os.getpid()
_REGISTRY_GUARD = threading.RLock()
_ACTIVE_LOCKS_GUARD = threading.RLock()
_THREAD_STATE = threading.local()


@dataclasses.dataclass(slots=True)
class _LockState:
    thread_lock: Any = dataclasses.field(default_factory=threading.RLock)
    process_id: int = dataclasses.field(default_factory=os.getpid)
    retainers: set[object] = dataclasses.field(default_factory=set)


@dataclasses.dataclass(frozen=True, slots=True)
class _LockResource:
    key: str
    path: pathlib.Path


@dataclasses.dataclass(slots=True)
class _AcquiredResource:
    resource: _LockResource
    state: _LockState | None = None
    retention_token: object = dataclasses.field(default_factory=object)
    thread_depth_before: int | None = None
    thread_lock_acquired: bool = False
    file_lock: portalocker.Lock | None = None
    file_lock_acquired: bool = False


@dataclasses.dataclass(eq=False, slots=True)
class _MutexObligation:
    lock: Any
    phase: str = "acquiring"


_LOCK_STATES: dict[str, _LockState] = {}
_ACTIVE_FILE_LOCKS: dict[int, portalocker.Lock] = {}


@contextlib.contextmanager
def index_guard(config: app_config.AppConfig) -> Iterator[pathlib.Path]:
    """Serialize lifecycle work for both manifest and storage target.

    The guarantee covers processes owned by the same user on one host. Qdrant
    deployments with lifecycle operations on multiple hosts need external
    distributed coordination.

    Args:
      config: Settings identifying the local file or Qdrant collection.

    Yields:
      The canonical manifest target while this thread owns both lifecycle
      resources.

    Raises:
      ConfigurationError: If either target is invalid, the lock cannot be
        acquired before the finite timeout, or a lock artifact cannot be
        opened or updated.
    """
    _ensure_current_process()
    resources, manifest_path = _lock_resources(config)
    _validate_nested_order(resources)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    acquired: list[_AcquiredResource] = []
    try:
        for resource in resources:
            frame = _AcquiredResource(resource=resource)
            acquired.append(frame)
            _retain_lock_state(frame)
            _acquire_resource(frame, deadline)
        yield manifest_path
    finally:
        active_error = sys.exception()
        release_error: BaseException | None = None
        for frame in reversed(acquired):
            try:
                _release_resource(frame)
            except BaseException as exc:
                if release_error is None:
                    release_error = exc
                with contextlib.suppress(BaseException):
                    _release_resource(frame)
        if active_error is None and release_error is not None:
            raise release_error


def _lock_resources(
    config: app_config.AppConfig,
) -> tuple[tuple[_LockResource, ...], pathlib.Path]:
    # Imported lazily to avoid an ingest -> index_lock -> ingest cycle.
    from local_docs_rag_agent.rag import ingest, manifest

    ingest.validate_storage_target(config)
    manifest_path = manifest.canonical_path(config.ingest_manifest_path)
    manifest_resource = _local_resource("manifest", manifest_path)
    if config.vector_backend == "local":
        index_path = config.index_path.resolve()
        if _paths_alias(index_path, manifest_path):
            raise exceptions.ConfigurationError(
                "INDEX_PATH and INGEST_MANIFEST_PATH must identify different "
                "files",
                action_hint=(
                    "Choose separate canonical paths for chunk storage and "
                    "the ingest manifest, then retry."
                ),
            )
        storage_resource = _local_resource("local-store", index_path)
    else:
        storage_key = _resource_key(
            "remote-store", ingest.storage_identity(config)
        )
        storage_resource = _LockResource(
            key=storage_key,
            path=_REMOTE_LOCK_ROOT / f"{storage_key}.lock",
        )
    return (
        tuple(
            sorted((manifest_resource, storage_resource), key=lambda x: x.key)
        ),
        manifest_path,
    )


def _paths_alias(first: pathlib.Path, second: pathlib.Path) -> bool:
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def _local_resource(kind: str, target: pathlib.Path) -> _LockResource:
    key = _resource_key(kind, str(target))
    return _LockResource(
        key=key,
        path=target.parent / f".local-docs-rag-agent-{kind}-{key}.lock",
    )


def _resource_key(kind: str, target: str) -> str:
    value = f"{kind}:{target}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _acquire_resource(
    frame: _AcquiredResource,
    deadline: float,
) -> None:
    depths = _thread_depths()
    key = frame.resource.key
    depth_before = depths.get(key, 0)
    frame.thread_depth_before = depth_before
    if depth_before:
        depths[key] = depths[key] + 1
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout_error()
    assert frame.state is not None
    try:
        if not frame.state.thread_lock.acquire(timeout=remaining):
            raise _timeout_error()
        frame.thread_lock_acquired = True
    except BaseException:
        if not frame.thread_lock_acquired:
            _release_unpublished_thread_lock(frame)
        raise
    try:
        frame.resource.path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _lock_io_error() from exc
    frame.file_lock = portalocker.Lock(
        frame.resource.path,
        mode="a",
        timeout=0.0,
        check_interval=0.05,
        fail_when_locked=True,
    )
    _acquire_file_lock(frame, deadline)
    depths[key] = 1


def _acquire_file_lock(
    frame: _AcquiredResource,
    deadline: float,
) -> None:
    assert frame.file_lock is not None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _timeout_error()
        try:
            with _active_locks_section():
                frame.file_lock.acquire(
                    timeout=0.0,
                    check_interval=0.05,
                    fail_when_locked=True,
                )
                frame.file_lock_acquired = True
                _ACTIVE_FILE_LOCKS[id(frame.file_lock)] = frame.file_lock
        except portalocker.exceptions.LockException:
            _rollback_file_lock_acquisition(frame)
            time.sleep(min(0.05, remaining))
            continue
        except OSError as exc:
            _rollback_file_lock_acquisition(frame)
            raise _lock_io_error() from exc
        except BaseException:
            _rollback_file_lock_acquisition(frame)
            raise
        return


def _release_unpublished_thread_lock(frame: _AcquiredResource) -> None:
    assert frame.state is not None
    with contextlib.suppress(BaseException):
        frame.state.thread_lock.release()


def _rollback_file_lock_acquisition(frame: _AcquiredResource) -> None:
    file_lock = frame.file_lock
    if file_lock is None:
        return
    registered = False
    try:
        with _active_locks_section():
            registered = id(file_lock) in _ACTIVE_FILE_LOCKS
            _ACTIVE_FILE_LOCKS.pop(id(file_lock), None)
    except BaseException:
        pass
    file_handle = getattr(file_lock, "fh", None)
    if frame.file_lock_acquired or registered or file_handle is not None:
        try:
            file_lock.release()
        except BaseException:
            file_handle = getattr(file_lock, "fh", None)
            if file_handle is None:
                frame.file_lock_acquired = False
                return
            with contextlib.suppress(BaseException):
                file_handle.close()
            with contextlib.suppress(BaseException):
                file_lock.fh = None
    frame.file_lock_acquired = False


def _release_resource(frame: _AcquiredResource) -> None:
    release_error: BaseException | None = None
    for release in (
        _restore_thread_depth,
        _release_file_lock,
        _release_thread_lock,
        _release_lock_state,
    ):
        try:
            release(frame)
        except BaseException as exc:
            if release_error is None:
                release_error = exc
    if release_error is not None:
        raise release_error


def _restore_thread_depth(frame: _AcquiredResource) -> None:
    key = frame.resource.key
    depths = _thread_depths()
    depth_before = frame.thread_depth_before
    if depth_before is None or depths.get(key, 0) <= depth_before:
        return
    if depth_before:
        depths[key] = depth_before
    else:
        depths.pop(key, None)


def _release_file_lock(frame: _AcquiredResource) -> None:
    if not _thread_depth_restored(frame):
        return
    file_lock = frame.file_lock
    if file_lock is None:
        return
    with _active_locks_section():
        registered = id(file_lock) in _ACTIVE_FILE_LOCKS
    if not frame.file_lock_acquired and not registered:
        return
    try:
        with _active_locks_section():
            file_lock.release()
            _ACTIVE_FILE_LOCKS.pop(id(file_lock), None)
            frame.file_lock_acquired = False
    except (OSError, portalocker.exceptions.LockException) as exc:
        _force_release_file_lock(frame)
        raise _lock_io_error() from exc
    except BaseException:
        _force_release_file_lock(frame)
        raise


def _force_release_file_lock(frame: _AcquiredResource) -> None:
    file_lock = frame.file_lock
    if file_lock is None:
        return
    with contextlib.suppress(BaseException), _active_locks_section():
        released = False
        try:
            file_lock.release()
        except BaseException:
            file_handle = getattr(file_lock, "fh", None)
            if file_handle is None:
                released = True
            else:
                try:
                    file_handle.close()
                except BaseException:
                    pass
                else:
                    released = True
                    with contextlib.suppress(BaseException):
                        file_lock.fh = None
        else:
            released = True
        if released:
            _ACTIVE_FILE_LOCKS.pop(id(file_lock), None)
            frame.file_lock_acquired = False


def _release_thread_lock(frame: _AcquiredResource) -> None:
    if not _thread_depth_restored(frame) or not frame.thread_lock_acquired:
        return
    assert frame.state is not None
    try:
        frame.state.thread_lock.release()
        frame.thread_lock_acquired = False
    except BaseException:
        _force_release_thread_lock(frame)
        raise


def _force_release_thread_lock(frame: _AcquiredResource) -> None:
    if not frame.thread_lock_acquired or frame.state is None:
        return
    try:
        frame.state.thread_lock.release()
    except RuntimeError as exc:
        if _already_released_error(exc):
            frame.thread_lock_acquired = False
    except BaseException:
        return
    else:
        frame.thread_lock_acquired = False


def _already_released_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("un-acquired", "unowned", "unlocked lock")
    )


def _retain_lock_state(frame: _AcquiredResource) -> None:
    key = frame.resource.key
    with _registry_section():
        state = _LOCK_STATES.get(key)
        if state is None or state.process_id != _PROCESS_ID:
            state = _LockState()
        frame.state = state
        state.retainers.add(frame.retention_token)
        _LOCK_STATES[key] = state
        return


def _release_lock_state(frame: _AcquiredResource) -> None:
    if (
        not _thread_depth_restored(frame)
        or frame.thread_lock_acquired
        or frame.file_lock_acquired
    ):
        return
    key = frame.resource.key
    state = frame.state
    if state is None:
        return
    with _registry_section():
        if _LOCK_STATES.get(key) is not state:
            return
        state.retainers.discard(frame.retention_token)
        if not state.retainers:
            _LOCK_STATES.pop(key, None)


@contextlib.contextmanager
def _registry_section() -> Iterator[None]:
    with _mutex_section(_REGISTRY_GUARD):
        yield


@contextlib.contextmanager
def _active_locks_section() -> Iterator[None]:
    with _mutex_section(_ACTIVE_LOCKS_GUARD):
        yield


@contextlib.contextmanager
def _mutex_section(lock: Any) -> Iterator[None]:
    key = id(lock)
    depths = _mutex_depths()
    depth_before = depths.get(key, 0)
    if depth_before:
        depths[key] = depth_before + 1
        try:
            yield
        finally:
            _restore_mutex_depth(key, depth_before)
        return

    _drain_mutex_obligations(lock)
    obligation = _MutexObligation(lock=lock)
    body_error: BaseException | None = None
    try:
        _mutex_obligations().append(obligation)
        body_error = None
        try:
            if not lock.acquire():
                raise RuntimeError("internal lifecycle mutex was not acquired")
            obligation.phase = "active"
        except BaseException:
            obligation.phase = "cleanup"
            raise
        depths[key] = 1
        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
    finally:
        active_error = sys.exception()
        _restore_mutex_depth(key, depth_before)
        obligation.phase = "cleanup"
        try:
            _release_mutex_obligation(obligation)
        except BaseException:
            if active_error is None and body_error is None:
                raise


def _release_mutex_obligation(obligation: _MutexObligation) -> None:
    if not _mutex_obligation_pending(obligation):
        return
    first_error: BaseException | None = None
    for _ in range(2):
        try:
            obligation.lock.release()
        except RuntimeError as exc:
            if _already_released_error(exc):
                _resolve_mutex_obligation(obligation)
                break
            if first_error is None:
                first_error = exc
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        else:
            _resolve_mutex_obligation(obligation)
            break
    if first_error is not None:
        raise first_error


def _drain_mutex_obligations(lock: Any) -> None:
    for obligation in tuple(_mutex_obligations()):
        if obligation.lock is lock and obligation.phase == "cleanup":
            _release_mutex_obligation(obligation)


def _resolve_mutex_obligation(obligation: _MutexObligation) -> None:
    obligations = _mutex_obligations()
    for index, pending in enumerate(obligations):
        if pending is obligation:
            obligations.pop(index)
            return


def _mutex_obligation_pending(obligation: _MutexObligation) -> bool:
    return any(pending is obligation for pending in _mutex_obligations())


def _mutex_obligations() -> list[_MutexObligation]:
    obligations = getattr(_THREAD_STATE, "mutex_obligations", None)
    if obligations is None:
        obligations = []
        _THREAD_STATE.mutex_obligations = obligations
    return obligations


def _mutex_depths() -> dict[int, int]:
    depths = getattr(_THREAD_STATE, "mutex_depths", None)
    if depths is None:
        depths = {}
        _THREAD_STATE.mutex_depths = depths
    return depths


def _restore_mutex_depth(key: int, depth_before: int) -> None:
    depths = _mutex_depths()
    if depth_before:
        depths[key] = depth_before
    else:
        depths.pop(key, None)


def _validate_nested_order(resources: tuple[_LockResource, ...]) -> None:
    held = {key for key, depth in _thread_depths().items() if depth > 0}
    if not held:
        return
    requested = {resource.key for resource in resources}
    overlap = held & requested
    new = requested - held
    if overlap and new:
        raise _unsafe_nested_error(
            "partially overlaps resources held by the current thread"
        )
    if new and min(new) < max(held):
        raise _unsafe_nested_error("would acquire resources out of order")


def _unsafe_nested_error(reason: str) -> exceptions.ConfigurationError:
    return exceptions.ConfigurationError(
        f"Unsafe nested index lifecycle guard: {reason}",
        action_hint=(
            "Exit the current index guard before entering a configuration "
            "with different lifecycle resources, or acquire disjoint "
            "configurations in canonical resource order."
        ),
    )


def _thread_depths() -> dict[str, int]:
    depths = getattr(_THREAD_STATE, "depths", None)
    if depths is None:
        depths = {}
        _THREAD_STATE.depths = depths
    return depths


def _thread_depth_restored(frame: _AcquiredResource) -> bool:
    depth_before = frame.thread_depth_before
    if depth_before is None:
        return True
    return _thread_depths().get(frame.resource.key, 0) <= depth_before


def _timeout_error() -> exceptions.ConfigurationError:
    return exceptions.ConfigurationError(
        "Timed out waiting for the index lifecycle lock",
        action_hint=(
            "Retry after the active ingest or retrieval finishes. If no "
            "process is active, verify local permissions and external "
            "coordination before retrying."
        ),
    )


def _lock_io_error() -> exceptions.ConfigurationError:
    return exceptions.ConfigurationError(
        "Could not open or update the index lifecycle lock",
        action_hint=(
            "Check permissions for the index directory or per-user lock "
            "cache, then retry."
        ),
    )


def _ensure_current_process() -> None:
    if os.getpid() != _PROCESS_ID:
        _reset_after_fork()


def _before_fork() -> None:
    _REGISTRY_GUARD.acquire()
    _ACTIVE_LOCKS_GUARD.acquire()


def _after_fork_parent() -> None:
    _ACTIVE_LOCKS_GUARD.release()
    _REGISTRY_GUARD.release()


def _reset_after_fork() -> None:
    global _ACTIVE_FILE_LOCKS
    global _ACTIVE_LOCKS_GUARD
    global _LOCK_STATES
    global _PROCESS_ID
    global _REGISTRY_GUARD
    global _THREAD_STATE

    for inherited_lock in _ACTIVE_FILE_LOCKS.values():
        file_handle = inherited_lock.fh
        if file_handle is None:
            continue
        with contextlib.suppress(OSError):
            file_handle.close()
        inherited_lock.fh = None
    _PROCESS_ID = os.getpid()
    _REGISTRY_GUARD = threading.RLock()
    _ACTIVE_LOCKS_GUARD = threading.RLock()
    _THREAD_STATE = threading.local()
    _LOCK_STATES = {}
    _ACTIVE_FILE_LOCKS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_reset_after_fork,
    )
