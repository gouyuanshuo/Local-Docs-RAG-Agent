"""Immutable, validated runtime configuration for every delivery mechanism.

:class:`AppConfig` is the single snapshot of settings that the API, the CLI, the
ingest pipeline, and the eval harness all read from. It is frozen so that a
request or an eval run can derive a variant with
:meth:`AppConfig.with_overrides` without mutating state another caller is still
using, and every variant is re-validated on construction.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, cast

from local_docs_rag_agent import constants, env, exceptions

DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"

LLM_API_KEY_VARIABLES = ("LLM_API_KEY", "OPENAI_API_KEY")
LLM_MODEL_VARIABLES = ("LLM_MODEL", "OPENAI_MODEL")
EMBEDDING_API_KEY_VARIABLES = (
    "EMBEDDING_API_KEY",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
)
EMBEDDING_BASE_URL_VARIABLES = ("EMBEDDING_BASE_URL", "LLM_BASE_URL")


@dataclasses.dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated snapshot of the settings that drive one process or request."""

    llm_provider: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    llm_api_style: constants.ApiStyleName
    embedding_provider: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    embedding_dimensions: int | None
    embedding_batch_size: int
    embedding_max_retries: int
    embedding_retry_backoff_ms: int
    docs_dir: pathlib.Path
    docs_exclude_patterns: list[str]
    index_path: pathlib.Path
    ingest_manifest_path: pathlib.Path
    eval_path: pathlib.Path
    agent_runtime: constants.RuntimeName
    agents_max_turns: int
    vector_backend: constants.VectorBackendName
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_timeout_s: int
    external_http_trust_env: bool
    top_k: int
    chunk_strategy: constants.ChunkStrategyName
    chunk_size: int
    chunk_overlap: int
    retrieval_strategy: constants.RetrievalStrategyName
    retrieval_candidate_k: int
    rrf_k: int
    reranker: constants.RerankerName
    rerank_candidate_k: int
    rerank_model: str | None

    def __post_init__(self) -> None:
        # Every instance is re-validated, because `with_overrides` can build a
        # variant from values that never passed through the environment readers.
        env.require_choice(
            "LLM_API_STYLE", self.llm_api_style, constants.API_STYLES
        )
        env.require_choice(
            "AGENT_RUNTIME", self.agent_runtime, constants.AGENT_RUNTIMES
        )
        env.require_choice(
            "VECTOR_BACKEND", self.vector_backend, constants.VECTOR_BACKENDS
        )
        env.require_choice(
            "CHUNK_STRATEGY", self.chunk_strategy, constants.CHUNK_STRATEGIES
        )
        env.require_choice(
            "RETRIEVAL_STRATEGY",
            self.retrieval_strategy,
            constants.RETRIEVAL_STRATEGIES,
        )
        env.require_choice("RERANKER", self.reranker, constants.RERANKERS)
        env.require_positive("EMBEDDING_BATCH_SIZE", self.embedding_batch_size)
        env.require_non_negative(
            "EMBEDDING_MAX_RETRIES", self.embedding_max_retries
        )
        env.require_non_negative(
            "EMBEDDING_RETRY_BACKOFF_MS", self.embedding_retry_backoff_ms
        )
        env.require_positive("AGENTS_MAX_TURNS", self.agents_max_turns)
        env.require_positive("QDRANT_TIMEOUT_S", self.qdrant_timeout_s)
        env.require_positive("TOP_K", self.top_k)
        env.require_positive("CHUNK_SIZE", self.chunk_size)
        env.require_non_negative("CHUNK_OVERLAP", self.chunk_overlap)
        env.require_positive(
            "RETRIEVAL_CANDIDATE_K", self.retrieval_candidate_k
        )
        env.require_positive("RRF_K", self.rrf_k)
        env.require_positive("RERANK_CANDIDATE_K", self.rerank_candidate_k)
        env.require_non_empty("LLM_MODEL", self.llm_model)
        env.require_non_empty("EMBEDDING_MODEL", self.embedding_model)
        env.require_non_empty("QDRANT_COLLECTION", self.qdrant_collection)
        if self.chunk_overlap >= self.chunk_size:
            raise exceptions.ConfigurationError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(overlap={self.chunk_overlap}, size={self.chunk_size})"
            )
        if self.embedding_dimensions is not None:
            env.require_positive(
                "EMBEDDING_DIMENSIONS", self.embedding_dimensions
            )

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build a configuration from the process environment and project `.env`."""

        env.load_project_dotenv()
        llm_provider = env.env_token("LLM_PROVIDER", "openai")
        embedding_provider = env.env_token("EMBEDDING_PROVIDER", llm_provider)
        return cls(
            llm_provider=llm_provider,
            llm_api_key=env.env_first_optional(LLM_API_KEY_VARIABLES),
            llm_base_url=env.env_optional_text("LLM_BASE_URL"),
            llm_model=env.env_first_text(
                LLM_MODEL_VARIABLES, DEFAULT_LLM_MODEL
            ),
            llm_api_style=env.env_choice(
                "LLM_API_STYLE",
                constants.DEFAULT_API_STYLE,
                constants.API_STYLES,
            ),
            embedding_provider=embedding_provider,
            embedding_api_key=env.env_first_optional(
                EMBEDDING_API_KEY_VARIABLES
            ),
            embedding_base_url=env.env_first_optional(
                EMBEDDING_BASE_URL_VARIABLES
            ),
            embedding_model=env.env_text(
                "EMBEDDING_MODEL",
                default_embedding_model(embedding_provider),
            ),
            embedding_dimensions=env.env_optional_int("EMBEDDING_DIMENSIONS"),
            embedding_batch_size=env.env_int("EMBEDDING_BATCH_SIZE", 10),
            embedding_max_retries=env.env_int("EMBEDDING_MAX_RETRIES", 2),
            embedding_retry_backoff_ms=env.env_int(
                "EMBEDDING_RETRY_BACKOFF_MS", 800
            ),
            docs_dir=env.env_path("DOCS_DIR", "docs"),
            docs_exclude_patterns=env.env_list("DOCS_EXCLUDE_PATTERNS"),
            index_path=env.env_path("INDEX_PATH", "data/index/chunks.jsonl"),
            ingest_manifest_path=env.env_path(
                "INGEST_MANIFEST_PATH",
                "data/index/ingest_manifest.json",
            ),
            eval_path=env.env_path("EVAL_PATH", "data/evals/sample_eval.jsonl"),
            agent_runtime=env.env_choice(
                "AGENT_RUNTIME",
                constants.DEFAULT_AGENT_RUNTIME,
                constants.AGENT_RUNTIMES,
            ),
            agents_max_turns=env.env_int("AGENTS_MAX_TURNS", 6),
            vector_backend=env.env_choice(
                "VECTOR_BACKEND",
                constants.DEFAULT_VECTOR_BACKEND,
                constants.VECTOR_BACKENDS,
            ),
            qdrant_url=env.env_optional_text("QDRANT_URL"),
            qdrant_api_key=env.env_optional_text("QDRANT_API_KEY"),
            qdrant_collection=env.env_text(
                "QDRANT_COLLECTION", "local-docs-rag"
            ),
            qdrant_timeout_s=env.env_int("QDRANT_TIMEOUT_S", 30),
            external_http_trust_env=env.env_bool(
                "EXTERNAL_HTTP_TRUST_ENV", True
            ),
            top_k=env.env_int("TOP_K", 4),
            chunk_strategy=env.env_choice(
                "CHUNK_STRATEGY",
                constants.DEFAULT_CHUNK_STRATEGY,
                constants.CHUNK_STRATEGIES,
            ),
            chunk_size=env.env_int("CHUNK_SIZE", 800),
            chunk_overlap=env.env_int("CHUNK_OVERLAP", 120),
            retrieval_strategy=env.env_choice(
                "RETRIEVAL_STRATEGY",
                constants.DEFAULT_RETRIEVAL_STRATEGY,
                constants.RETRIEVAL_STRATEGIES,
            ),
            retrieval_candidate_k=env.env_int(
                "RETRIEVAL_CANDIDATE_K", constants.DEFAULT_RETRIEVAL_CANDIDATE_K
            ),
            rrf_k=env.env_int("RRF_K", constants.DEFAULT_RRF_K),
            reranker=env.env_choice(
                "RERANKER", constants.DEFAULT_RERANKER, constants.RERANKERS
            ),
            rerank_candidate_k=env.env_int(
                "RERANK_CANDIDATE_K", constants.DEFAULT_RERANK_CANDIDATE_K
            ),
            rerank_model=env.env_optional_text("RERANK_MODEL"),
        )

    def with_runtime(self, runtime: str | None) -> AppConfig:
        """Return a variant that answers with `runtime`, or `self` when it is unset."""

        if runtime is None:
            return self
        # `__post_init__` rejects an unsupported name, so the cast cannot
        # smuggle one in.
        return self.with_overrides(
            agent_runtime=cast(constants.RuntimeName, runtime.strip().lower())
        )

    def with_overrides(self, **overrides: Any) -> AppConfig:
        """Return a re-validated variant with the named fields replaced."""

        known_fields = {field.name for field in dataclasses.fields(self)}
        unknown_fields = sorted(set(overrides) - known_fields)
        if unknown_fields:
            raise exceptions.ConfigurationError(
                f"Unknown AppConfig override(s): {', '.join(unknown_fields)}"
            )
        return dataclasses.replace(self, **overrides)


def default_embedding_model(provider: str) -> str:
    """Return the embedding model to use when `EMBEDDING_MODEL` is not configured."""

    if provider == "qwen":
        return DEFAULT_QWEN_EMBEDDING_MODEL
    return DEFAULT_OPENAI_EMBEDDING_MODEL
