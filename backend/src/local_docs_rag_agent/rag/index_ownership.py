"""Fail-closed ownership tokens for disposable Qdrant index cleanup."""

from __future__ import annotations

import dataclasses
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
    """Opaque proof that this process claimed one absent Qdrant collection."""

    _nonce: str
    _storage_identity: str
    _collection_name: str


_CLAIMS_GUARD = threading.Lock()
_CLAIMS: dict[str, tuple[str, str]] = {}


def claim_qdrant_index_ownership(
    config: app_config.AppConfig,
) -> QdrantIndexOwnership:
    """Claim cleanup ownership of an absent configured Qdrant collection.

    Args:
      config: A Qdrant configuration naming a disposable collection.

    Returns:
      An opaque process-local token required for exact-target cleanup.

    Raises:
      ConfigurationError: If the backend is not Qdrant, the collection already
        exists, or this process already holds a claim for the same target.
      VectorStoreError: If collection readiness cannot be checked.
    """
    _require_qdrant(config)
    with index_lock.index_guard(config):
        store = _qdrant_store(config)
        if store.collection_exists():
            raise exceptions.ConfigurationError(
                "Cannot claim Qdrant index ownership: collection already "
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
        return QdrantIndexOwnership(
            _nonce=nonce,
            _storage_identity=storage_identity,
            _collection_name=collection_name,
        )


def delete_owned_qdrant_index(
    config: app_config.AppConfig,
    ownership: QdrantIndexOwnership,
) -> None:
    """Delete exactly the disposable Qdrant collection `ownership` claimed.

    Args:
      config: The same target configuration used when claiming ownership.
      ownership: Opaque token returned by `claim_qdrant_index_ownership`.

    Raises:
      ConfigurationError: If the backend, target identity, collection, token,
        or process-local claim does not match exactly.
      VectorStoreError: If Qdrant readiness or deletion fails.
    """
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
