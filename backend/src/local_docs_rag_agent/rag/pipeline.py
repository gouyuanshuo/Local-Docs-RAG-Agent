"""Composes the retrieval stages into the one call the rest of the backend makes.

Retrieval is no longer a single lookup. A store ranks the corpus with the
configured strategy, and a reranker may then reorder a wider candidate window
before anything is answered from it. Both stages are selected by configuration,
and both can degrade.

Composing them here rather than inside a store keeps each piece answerable for
one thing: `ChunkStore` stays a storage-and-search contract that a new backend
can satisfy without knowing that reranking exists, and `Reranker` stays
implementable without knowing which backend produced its candidates. It also
means the second stage cannot be skipped by accident — every caller that
retrieves for an answer goes through :func:`retrieve` and gets both stages and
both statuses.

The candidate depth comes from the reranker rather than from the caller, because
only the reranker knows how much it is willing to read. With reranking off that
depth is exactly `top_k`, so the disabled path issues the same query the project
issued before this module existed.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import models
from local_docs_rag_agent.rag import llm_rerank, rerank, store_factory


def build_reranker(config: app_config.AppConfig) -> rerank.Reranker:
    """Return the reranker selected by `RERANKER`.

    The LLM reranker reuses the chat credentials, endpoint, and API style, so an
    OpenAI-compatible deployment that answers questions can rerank for them too
    without a second set of settings. `RERANK_MODEL` overrides only the model,
    which is what makes a small, cheap ranking model usable alongside a larger
    answering one.
    """

    if config.reranker == "llm":
        return llm_rerank.LlmReranker(
            api_key=config.llm_api_key,
            model=config.rerank_model or config.llm_model,
            candidate_k=config.rerank_candidate_k,
            base_url=config.llm_base_url,
            api_style=config.llm_api_style,
            trust_env=config.external_http_trust_env,
        )
    return rerank.IdentityReranker()


def retrieve(
    config: app_config.AppConfig, query: str, top_k: int | None = None
) -> models.RetrievalOutcome:
    """Search the configured store, rerank the candidates, and report both stages."""

    limit = config.top_k if top_k is None else top_k
    store = store_factory.build_store(config)
    reranker = build_reranker(config)
    candidates = store.search(
        query=query, top_k=reranker.candidate_depth(limit)
    )
    hits = reranker.rerank(query=query, hits=candidates, top_k=limit)
    # Both statuses are read after the work, because a provider reports how it
    # actually behaved on this call rather than how it was configured.
    return models.RetrievalOutcome(
        hits=hits,
        embedding_status=store.embedding_status,
        reranker_status=reranker.status,
    )
