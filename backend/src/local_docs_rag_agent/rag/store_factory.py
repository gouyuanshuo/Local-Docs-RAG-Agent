"""Builds the configured chunk store.

This is the only place that turns `VECTOR_BACKEND` into a concrete store, so ingest,
retrieval, the agent tools, and the connectivity scripts all reach the same backend
with the same embedding provider. Adding a backend means adding a branch here and an
implementation of `ChunkStore` — nothing else needs to know it exists.
"""

from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ConfigurationError
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.providers.factory import build_embedding_provider
from local_docs_rag_agent.rag.base import ChunkStore
from local_docs_rag_agent.rag.local_store import LocalJsonlChunkStore
from local_docs_rag_agent.rag.qdrant_store import QdrantChunkStore
from local_docs_rag_agent.rag.retrieval import RetrievalSettings


def build_store(
    config: AppConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> ChunkStore:
    """Return the chunk store selected by `config`.

    Callers that already hold an embedding provider should pass it, so a single ingest
    run reports one coherent provider status instead of building a second client whose
    health is tracked separately.
    """

    provider = embedding_provider or build_embedding_provider(config)
    settings = retrieval_settings(config)
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise ConfigurationError("QDRANT_URL must be set when VECTOR_BACKEND=qdrant")
        return QdrantChunkStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            timeout_s=config.qdrant_timeout_s,
            embedding_provider=provider,
            trust_env=config.external_http_trust_env,
            settings=settings,
        )
    return LocalJsonlChunkStore(
        config.index_path,
        embedding_provider=provider,
        settings=settings,
    )


def retrieval_settings(config: AppConfig) -> RetrievalSettings:
    """Project the ranking knobs out of `config` for the stores to consume."""

    return RetrievalSettings(
        strategy=config.retrieval_strategy,
        candidate_k=config.retrieval_candidate_k,
        rrf_k=config.rrf_k,
    )
