from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from local_docs_rag_agent.exceptions import ConfigurationError

SUPPORTED_AGENT_RUNTIMES = frozenset({"basic", "agents_sdk"})
SUPPORTED_API_STYLES = frozenset({"responses", "chat_completions"})
SUPPORTED_CHUNK_STRATEGIES = frozenset({"fixed", "paragraph", "markdown"})
SUPPORTED_VECTOR_BACKENDS = frozenset({"local", "qdrant"})


def _load_dotenv() -> None:
    # Load a project-local .env if present, without overriding already-exported variables.
    load_dotenv(override=False)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _list_env(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_embedding_model(provider: str) -> str:
    if provider == "qwen":
        return "text-embedding-v4"
    return "text-embedding-3-small"


@dataclass(frozen=True, slots=True)
class AppConfig:
    llm_provider: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    llm_api_style: str
    embedding_provider: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    embedding_dimensions: int | None
    embedding_batch_size: int
    embedding_max_retries: int
    embedding_retry_backoff_ms: int
    docs_dir: Path
    docs_exclude_patterns: list[str]
    index_path: Path
    ingest_manifest_path: Path
    eval_path: Path
    agent_runtime: str
    agents_max_turns: int
    vector_backend: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_timeout_s: int
    external_http_trust_env: bool
    top_k: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        _require_choice("LLM_API_STYLE", self.llm_api_style, SUPPORTED_API_STYLES)
        _require_choice("AGENT_RUNTIME", self.agent_runtime, SUPPORTED_AGENT_RUNTIMES)
        _require_choice("VECTOR_BACKEND", self.vector_backend, SUPPORTED_VECTOR_BACKENDS)
        _require_choice("CHUNK_STRATEGY", self.chunk_strategy, SUPPORTED_CHUNK_STRATEGIES)
        _require_positive("EMBEDDING_BATCH_SIZE", self.embedding_batch_size)
        _require_non_negative("EMBEDDING_MAX_RETRIES", self.embedding_max_retries)
        _require_non_negative(
            "EMBEDDING_RETRY_BACKOFF_MS",
            self.embedding_retry_backoff_ms,
        )
        _require_positive("AGENTS_MAX_TURNS", self.agents_max_turns)
        _require_positive("QDRANT_TIMEOUT_S", self.qdrant_timeout_s)
        _require_positive("TOP_K", self.top_k)
        _require_positive("CHUNK_SIZE", self.chunk_size)
        _require_non_negative("CHUNK_OVERLAP", self.chunk_overlap)
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(overlap={self.chunk_overlap}, size={self.chunk_size})"
            )
        if self.embedding_dimensions is not None:
            _require_positive("EMBEDDING_DIMENSIONS", self.embedding_dimensions)
        if not self.llm_model.strip():
            raise ConfigurationError("LLM_MODEL must not be empty")
        if not self.embedding_model.strip():
            raise ConfigurationError("EMBEDDING_MODEL must not be empty")
        if not self.qdrant_collection.strip():
            raise ConfigurationError("QDRANT_COLLECTION must not be empty")

    @classmethod
    def from_env(cls) -> AppConfig:
        _load_dotenv()
        llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", llm_provider).strip().lower()
        return cls(
            llm_provider=llm_provider,
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_model=(os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"),
            llm_api_style=os.getenv("LLM_API_STYLE", "responses").strip().lower(),
            embedding_provider=embedding_provider,
            embedding_api_key=os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            embedding_base_url=(os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")),
            embedding_model=(
                os.getenv("EMBEDDING_MODEL") or _default_embedding_model(embedding_provider)
            ),
            embedding_dimensions=_optional_int_env("EMBEDDING_DIMENSIONS"),
            embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 10),
            embedding_max_retries=_int_env("EMBEDDING_MAX_RETRIES", 2),
            embedding_retry_backoff_ms=_int_env("EMBEDDING_RETRY_BACKOFF_MS", 800),
            docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
            docs_exclude_patterns=_list_env("DOCS_EXCLUDE_PATTERNS"),
            index_path=Path(os.getenv("INDEX_PATH", "data/index/chunks.jsonl")),
            ingest_manifest_path=Path(
                os.getenv(
                    "INGEST_MANIFEST_PATH",
                    "data/index/ingest_manifest.json",
                )
            ),
            eval_path=Path(os.getenv("EVAL_PATH", "data/evals/sample_eval.jsonl")),
            agent_runtime=os.getenv("AGENT_RUNTIME", "basic").strip().lower(),
            agents_max_turns=_int_env("AGENTS_MAX_TURNS", 6),
            vector_backend=os.getenv("VECTOR_BACKEND", "local").strip().lower(),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "local-docs-rag"),
            qdrant_timeout_s=_int_env("QDRANT_TIMEOUT_S", 30),
            external_http_trust_env=_bool_env("EXTERNAL_HTTP_TRUST_ENV", True),
            top_k=_int_env("TOP_K", 4),
            chunk_strategy=os.getenv("CHUNK_STRATEGY", "markdown").strip().lower(),
            chunk_size=_int_env("CHUNK_SIZE", 800),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 120),
        )

    def with_runtime(self, runtime: str | None) -> AppConfig:
        if runtime is None:
            return self
        return self.with_overrides(agent_runtime=runtime.strip().lower())

    def with_overrides(self, **overrides: Any) -> AppConfig:
        known_fields = {field.name for field in fields(self)}
        unknown_fields = sorted(set(overrides) - known_fields)
        if unknown_fields:
            raise ConfigurationError(f"Unknown AppConfig override(s): {', '.join(unknown_fields)}")
        return replace(self, **overrides)


def _require_choice(name: str, value: str, choices: frozenset[str]) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigurationError(f"{name} must be one of: {allowed}; got {value!r}")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0, got {value}")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ConfigurationError(f"{name} must be at least 0, got {value}")
