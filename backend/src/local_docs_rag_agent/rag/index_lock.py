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
_REGISTRY_GUARD = threading.Lock()
_ACTIVE_LOCKS_GUARD = threading.Lock()
_THREAD_STATE = threading.local()


@dataclasses.dataclass(slots=True)
class _LockState:
    thread_lock: Any = dataclasses.field(default_factory=threading.RLock)
    process_id: int = dataclasses.field(default_factory=os.getpid)
    users: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _LockResource:
    key: str
    path: pathlib.Path


@dataclasses.dataclass(slots=True)
class _AcquiredResource:
    resource: _LockResource
    state: _LockState
    thread_lock_acquired: bool = False
    file_lock: portalocker.Lock | None = None
    file_lock_acquired: bool = False


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
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    acquired: list[_AcquiredResource] = []
    try:
        for resource in resources:
            frame = _AcquiredResource(
                resource=resource,
                state=_retain_lock_state(resource.key),
            )
            acquired.append(frame)
            _acquire_resource(frame, deadline)
        yield manifest_path
    finally:
        active_error = sys.exception()
        release_error: BaseException | None = None
        for frame in reversed(acquired):
            try:
                _release_resource(frame, active_error or release_error)
            except BaseException as exc:
                release_error = exc
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
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not frame.state.thread_lock.acquire(timeout=remaining):
        raise _timeout_error()
    frame.thread_lock_acquired = True
    depths = _thread_depths()
    key = frame.resource.key
    if depths.get(key, 0) == 0:
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
    depths[key] = depths.get(key, 0) + 1


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
            with _ACTIVE_LOCKS_GUARD:
                frame.file_lock.acquire(
                    timeout=0.0,
                    check_interval=0.05,
                    fail_when_locked=True,
                )
                _ACTIVE_FILE_LOCKS[id(frame.file_lock)] = frame.file_lock
        except portalocker.exceptions.LockException:
            time.sleep(min(0.05, remaining))
            continue
        except OSError as exc:
            raise _lock_io_error() from exc
        frame.file_lock_acquired = True
        return


def _release_resource(
    frame: _AcquiredResource,
    active_error: BaseException | None,
) -> None:
    key = frame.resource.key
    depths = _thread_depths()
    if frame.thread_lock_acquired and key in depths:
        next_depth = depths[key] - 1
        if next_depth:
            depths[key] = next_depth
        else:
            depths.pop(key, None)
    try:
        if frame.file_lock is not None and frame.file_lock_acquired:
            with _ACTIVE_LOCKS_GUARD:
                try:
                    frame.file_lock.release()
                except (
                    OSError,
                    portalocker.exceptions.LockException,
                ) as exc:
                    if active_error is None:
                        raise _lock_io_error() from exc
                finally:
                    _ACTIVE_FILE_LOCKS.pop(id(frame.file_lock), None)
    finally:
        if frame.thread_lock_acquired:
            frame.state.thread_lock.release()
        _release_lock_state(key, frame.state)


def _retain_lock_state(key: str) -> _LockState:
    with _REGISTRY_GUARD:
        state = _LOCK_STATES.get(key)
        if state is None or state.process_id != _PROCESS_ID:
            state = _LockState()
            _LOCK_STATES[key] = state
        state.users += 1
        return state


def _release_lock_state(key: str, state: _LockState) -> None:
    with _REGISTRY_GUARD:
        if _LOCK_STATES.get(key) is not state:
            return
        state.users -= 1
        if state.users == 0:
            _LOCK_STATES.pop(key, None)


def _thread_depths() -> dict[str, int]:
    depths = getattr(_THREAD_STATE, "depths", None)
    if depths is None:
        depths = {}
        _THREAD_STATE.depths = depths
    return depths


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
    _REGISTRY_GUARD = threading.Lock()
    _ACTIVE_LOCKS_GUARD = threading.Lock()
    _THREAD_STATE = threading.local()
    _LOCK_STATES = {}
    _ACTIVE_FILE_LOCKS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_reset_after_fork,
    )
