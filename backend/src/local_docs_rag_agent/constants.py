"""Closed sets of option values shared by the domain, delivery, and RAG layers.

Every option that must agree across configuration parsing, HTTP request validation,
and CLI argument choices is declared once here. Adding a runtime, chunk strategy,
vector backend, or API style is therefore a single edit in this module plus the
implementation that handles it, instead of the same literal list repeated in
``config.py``, ``api/schemas.py``, ``cli.py``, and the command handlers.

The tuples are derived from the ``Literal`` aliases with :func:`typing.get_args` so
the static type and the runtime validation can never drift apart.
"""

from __future__ import annotations

from typing import Literal, get_args

RuntimeName = Literal["basic", "agents_sdk"]
ChunkStrategyName = Literal["fixed", "paragraph", "markdown"]
VectorBackendName = Literal["local", "qdrant"]
ApiStyleName = Literal["responses", "chat_completions"]
RetrievalStrategyName = Literal["blended", "dense", "lexical", "hybrid_rrf"]
RerankerName = Literal["none", "llm"]

AGENT_RUNTIMES: tuple[RuntimeName, ...] = get_args(RuntimeName)
CHUNK_STRATEGIES: tuple[ChunkStrategyName, ...] = get_args(ChunkStrategyName)
VECTOR_BACKENDS: tuple[VectorBackendName, ...] = get_args(VectorBackendName)
API_STYLES: tuple[ApiStyleName, ...] = get_args(ApiStyleName)
RETRIEVAL_STRATEGIES: tuple[RetrievalStrategyName, ...] = get_args(RetrievalStrategyName)
RERANKERS: tuple[RerankerName, ...] = get_args(RerankerName)

DEFAULT_AGENT_RUNTIME: RuntimeName = "basic"
DEFAULT_CHUNK_STRATEGY: ChunkStrategyName = "markdown"
DEFAULT_VECTOR_BACKEND: VectorBackendName = "local"
DEFAULT_API_STYLE: ApiStyleName = "responses"
# `blended` reproduces the ranking the project shipped before retrieval became
# selectable, so it stays the default and the baseline the others are compared to.
DEFAULT_RETRIEVAL_STRATEGY: RetrievalStrategyName = "blended"
DEFAULT_RETRIEVAL_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
# Reranking is off by default: it costs a model call per question, and every result
# recorded before it existed was produced without it.
DEFAULT_RERANKER: RerankerName = "none"
DEFAULT_RERANK_CANDIDATE_K = 20

__all__ = [
    "AGENT_RUNTIMES",
    "API_STYLES",
    "CHUNK_STRATEGIES",
    "DEFAULT_AGENT_RUNTIME",
    "DEFAULT_API_STYLE",
    "DEFAULT_CHUNK_STRATEGY",
    "DEFAULT_RERANKER",
    "DEFAULT_RERANK_CANDIDATE_K",
    "DEFAULT_RETRIEVAL_CANDIDATE_K",
    "DEFAULT_RETRIEVAL_STRATEGY",
    "DEFAULT_RRF_K",
    "DEFAULT_VECTOR_BACKEND",
    "RERANKERS",
    "RETRIEVAL_STRATEGIES",
    "VECTOR_BACKENDS",
    "ApiStyleName",
    "ChunkStrategyName",
    "RerankerName",
    "RetrievalStrategyName",
    "RuntimeName",
    "VectorBackendName",
]
