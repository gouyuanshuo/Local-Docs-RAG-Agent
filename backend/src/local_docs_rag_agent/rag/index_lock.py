"""Single-host lifecycle serialization for one configured index target.

The operating-system lock coordinates processes while an explicit reentrant
thread lock closes the gap left by advisory locks inside one process. Local
locks live beside the canonical index file; remote locks use a credential-free
identity in the user's cache. Only the owning thread may enter recursively.
Lock files are stable identity markers and may safely outlive a run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
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
_REGISTRY_GUARD = threading.Lock()
_THREAD_STATE = threading.local()


@dataclasses.dataclass(slots=True)
class _LockState:
    thread_lock: Any = dataclasses.field(default_factory=threading.RLock)


_LOCK_STATES: dict[str, _LockState] = {}


@contextlib.contextmanager
def index_guard(config: app_config.AppConfig) -> Iterator[None]:
    """Serialize index readiness, reads, and writes for one storage target.

    The guarantee covers processes owned by the same user on one host. Qdrant
    deployments with lifecycle operations on multiple hosts need external
    distributed coordination.

    Args:
      config: Settings identifying the local file or Qdrant collection.

    Yields:
      Control while this thread owns the target's lifecycle lock.

    Raises:
      ConfigurationError: If the lock cannot be acquired before the finite
        timeout or its lock artifact cannot be opened or updated.
    """
    key = _lock_key(config)
    state = _lock_state(key)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    if not state.thread_lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
        raise _timeout_error()
    depths = _thread_depths()
    reentrant = depths.get(key, 0) > 0
    file_lock: portalocker.Lock | None = None
    file_lock_acquired = False
    try:
        if not reentrant:
            lock_path = _lock_path(config, key)
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise _lock_io_error() from exc
            remaining = max(0.0, deadline - time.monotonic())
            file_lock = portalocker.Lock(
                lock_path,
                mode="a",
                timeout=remaining,
                check_interval=min(0.05, remaining or 0.05),
            )
            try:
                file_lock.acquire()
            except portalocker.exceptions.LockException:
                raise _timeout_error() from None
            except OSError as exc:
                raise _lock_io_error() from exc
            file_lock_acquired = True
        depths[key] = depths.get(key, 0) + 1
        try:
            yield
        finally:
            next_depth = depths[key] - 1
            if next_depth:
                depths[key] = next_depth
            else:
                depths.pop(key, None)
    finally:
        active_error = sys.exception()
        try:
            if file_lock is not None and file_lock_acquired:
                try:
                    file_lock.release()
                except (OSError, portalocker.exceptions.LockException) as exc:
                    if active_error is None:
                        raise _lock_io_error() from exc
        finally:
            state.thread_lock.release()


def _lock_key(config: app_config.AppConfig) -> str:
    if config.vector_backend == "local":
        target = f"local:{config.index_path.resolve()}"
    else:
        # Imported lazily to avoid an ingest -> index_lock -> ingest cycle.
        from local_docs_rag_agent.rag import ingest

        target = f"qdrant:{ingest.storage_identity(config)}"
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def _lock_path(
    config: app_config.AppConfig,
    key: str,
) -> pathlib.Path:
    if config.vector_backend == "local":
        target = config.index_path.resolve()
        return target.parent / f".local-docs-rag-agent-{key}.lock"
    return _REMOTE_LOCK_ROOT / f"{key}.lock"


def _lock_state(key: str) -> _LockState:
    with _REGISTRY_GUARD:
        return _LOCK_STATES.setdefault(key, _LockState())


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
