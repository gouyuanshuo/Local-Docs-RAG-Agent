"""Atomic initialization and fail-closed cleanup for disposable Qdrant."""

from __future__ import annotations

import dataclasses
import os
import threading
import uuid

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.core import exceptions
from local_docs_rag_agent.rag import (
    index_lock,
    ingest,
    qdrant_store,
    store_factory,
)


@dataclasses.dataclass(frozen=True, slots=True)
class QdrantIndexOwnership:
    """Opaque proof that this process initialized one Qdrant collection."""

    _nonce: str
    _storage_identity: str
    _collection_name: str


_CLAIMS_GUARD = threading.Lock()
_CLAIMS: dict[str, tuple[str, str]] = {}
_CLAIMS_PROCESS_ID = os.getpid()
_ORPHANED_COLLECTION_ATTRIBUTE = (
    "_local_docs_rag_agent_qdrant_orphaned_collection"
)


def initialize_owned_qdrant_index(
    config: app_config.AppConfig,
) -> QdrantIndexOwnership:
    """Create and claim an absent disposable Qdrant collection atomically.

    Args:
      config: A Qdrant configuration naming a disposable collection.

    Returns:
      An opaque process-local token required for exact-target cleanup after
      the first ingest has created the collection.

    Raises:
      ConfigurationError: If the backend is not Qdrant, the collection already
        exists, this process already owns the target, or first ingest cannot
        create a collection.
      ProviderUnavailableError: If first ingest cannot embed the documents.
      VectorStoreError: If collection readiness or first ingest fails.
    """
    _ensure_current_process()
    _require_qdrant(config)
    with index_lock.index_guard(config):
        store = _qdrant_store(config)
        if store.collection_exists():
            raise exceptions.ConfigurationError(
                "Cannot initialize Qdrant index ownership: collection already "
                "exists",
                action_hint=(
                    "Generate a new unique run-owned collection name. Never "
                    "claim the configured interactive collection."
                ),
            )
        storage_identity = ingest.storage_identity(config)
        collection_name = config.qdrant_collection
        with _CLAIMS_GUARD:
            if (storage_identity, collection_name) in _CLAIMS.values():
                raise exceptions.ConfigurationError(
                    "Qdrant index ownership is already claimed in this process"
                )
            nonce = uuid.uuid4().hex
            _CLAIMS[nonce] = (storage_identity, collection_name)
        ownership = QdrantIndexOwnership(
            _nonce=nonce,
            _storage_identity=storage_identity,
            _collection_name=collection_name,
        )
        try:
            ingest.ingest_documents(config)
            if not store.collection_exists():
                raise exceptions.ConfigurationError(
                    "First ingest did not create the owned Qdrant collection",
                    action_hint=(
                        "Use a non-empty comparison dataset and retry with a "
                        "new unique collection name."
                    ),
                )
        except BaseException as exc:
            cleanup_failed = False
            try:
                if store.collection_exists():
                    try:
                        store.delete_collection()
                    except BaseException:
                        cleanup_failed = True
            except BaseException:
                cleanup_failed = True
            finally:
                _discard_claim(ownership)
            if cleanup_failed:
                _mark_orphaned_collection(exc, collection_name)
            raise
        return ownership


def qdrant_orphaned_collection(exc: BaseException) -> str | None:
    """Return the exact Qdrant collection left by interrupted initialization.

    Args:
      exc: The original exception escaping owned Qdrant initialization.

    Returns:
      The exact collection requiring operator recovery, or None when cleanup
      succeeded or no collection was created.
    """
    value = getattr(exc, _ORPHANED_COLLECTION_ATTRIBUTE, None)
    return value if isinstance(value, str) and value else None


def delete_owned_qdrant_index(
    config: app_config.AppConfig,
    ownership: QdrantIndexOwnership,
) -> None:
    """Delete exactly the disposable Qdrant collection that was initialized.

    Args:
      config: The same target configuration used during initialization.
      ownership: Opaque token returned by
        `initialize_owned_qdrant_index`.

    Raises:
      ConfigurationError: If the backend, target identity, collection, token,
        or process-local claim does not match exactly.
      VectorStoreError: If Qdrant readiness or deletion fails.
    """
    _ensure_current_process()
    _require_qdrant(config)
    with index_lock.index_guard(config):
        storage_identity = ingest.storage_identity(config)
        expected = (storage_identity, config.qdrant_collection)
        token_value = (
            ownership._storage_identity,
            ownership._collection_name,
        )
        with _CLAIMS_GUARD:
            registered = _CLAIMS.get(ownership._nonce)
        if registered != expected or token_value != expected:
            raise exceptions.ConfigurationError(
                "Qdrant index ownership does not match the cleanup target",
                action_hint=(
                    "Refuse cleanup and retain the collection for targeted "
                    "operator review."
                ),
            )
        store = _qdrant_store(config)
        if store.collection_exists():
            store.delete_collection()
        with _CLAIMS_GUARD:
            _CLAIMS.pop(ownership._nonce, None)


def _discard_claim(ownership: QdrantIndexOwnership) -> None:
    with _CLAIMS_GUARD:
        _CLAIMS.pop(ownership._nonce, None)


def _mark_orphaned_collection(
    exc: BaseException,
    collection_name: str,
) -> None:
    setattr(exc, _ORPHANED_COLLECTION_ATTRIBUTE, collection_name)
    exc.add_note(
        "Qdrant initialization cleanup did not complete. Collection "
        f"{collection_name!r} may be orphaned; inspect that exact collection "
        "and delete only after verifying it belongs to the interrupted run."
    )


def _require_qdrant(config: app_config.AppConfig) -> None:
    if config.vector_backend != "qdrant":
        raise exceptions.ConfigurationError(
            "Qdrant index ownership requires VECTOR_BACKEND=qdrant"
        )


def _qdrant_store(
    config: app_config.AppConfig,
) -> qdrant_store.QdrantChunkStore:
    store = store_factory.build_store(config)
    if not isinstance(store, qdrant_store.QdrantChunkStore):
        raise exceptions.ConfigurationError(
            "Qdrant ownership requires a Qdrant collection store"
        )
    return store


def _ensure_current_process() -> None:
    if os.getpid() != _CLAIMS_PROCESS_ID:
        _reset_after_fork()


def _reset_after_fork() -> None:
    global _CLAIMS
    global _CLAIMS_GUARD
    global _CLAIMS_PROCESS_ID

    _CLAIMS_PROCESS_ID = os.getpid()
    _CLAIMS_GUARD = threading.Lock()
    _CLAIMS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
