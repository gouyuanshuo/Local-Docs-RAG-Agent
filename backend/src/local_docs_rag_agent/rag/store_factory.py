"""Builds the configured chunk store.

This is the only place that turns `VECTOR_BACKEND` into a concrete store, so
ingest, retrieval, the agent tools, and the connectivity scripts all reach the
same backend with the same embedding provider. Adding a backend means adding a
branch here and an implementation of `ChunkStore` — nothing else needs to know
it exists.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import exceptions
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import base as rag_base
from local_docs_rag_agent.rag import local_store, qdrant_store, retrieval


def build_store(
    config: app_config.AppConfig,
    embedding_provider: provider_base.EmbeddingProvider | None = None,
) -> rag_base.ChunkStore:
    """Return the chunk store selected by `config`.

    Callers that already hold an embedding provider should pass it, so a single
    ingest run reports one coherent provider status instead of building a second
    client whose health is tracked separately.
    """

    provider = embedding_provider or provider_factory.build_embedding_provider(
        config
    )
    settings = retrieval_settings(config)
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise exceptions.ConfigurationError(
                "QDRANT_URL must be set when VECTOR_BACKEND=qdrant"
            )
        return qdrant_store.QdrantChunkStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            timeout_s=config.qdrant_timeout_s,
            embedding_provider=provider,
            trust_env=config.external_http_trust_env,
            settings=settings,
        )
    return local_store.LocalJsonlChunkStore(
        config.index_path,
        embedding_provider=provider,
        settings=settings,
    )


def retrieval_settings(
    config: app_config.AppConfig,
) -> retrieval.RetrievalSettings:
    """Project the ranking knobs out of `config` for the stores to consume."""

    return retrieval.RetrievalSettings(
        strategy=config.retrieval_strategy,
        candidate_k=config.retrieval_candidate_k,
        rrf_k=config.rrf_k,
    )
