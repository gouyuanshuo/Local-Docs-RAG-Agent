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

AGENT_RUNTIMES: tuple[RuntimeName, ...] = get_args(RuntimeName)
CHUNK_STRATEGIES: tuple[ChunkStrategyName, ...] = get_args(ChunkStrategyName)
VECTOR_BACKENDS: tuple[VectorBackendName, ...] = get_args(VectorBackendName)
API_STYLES: tuple[ApiStyleName, ...] = get_args(ApiStyleName)

DEFAULT_AGENT_RUNTIME: RuntimeName = "basic"
DEFAULT_CHUNK_STRATEGY: ChunkStrategyName = "markdown"
DEFAULT_VECTOR_BACKEND: VectorBackendName = "local"
DEFAULT_API_STYLE: ApiStyleName = "responses"

__all__ = [
    "AGENT_RUNTIMES",
    "API_STYLES",
    "CHUNK_STRATEGIES",
    "DEFAULT_AGENT_RUNTIME",
    "DEFAULT_API_STYLE",
    "DEFAULT_CHUNK_STRATEGY",
    "DEFAULT_VECTOR_BACKEND",
    "VECTOR_BACKENDS",
    "ApiStyleName",
    "ChunkStrategyName",
    "RuntimeName",
    "VectorBackendName",
]
