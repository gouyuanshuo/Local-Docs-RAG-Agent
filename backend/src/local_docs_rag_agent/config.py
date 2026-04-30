from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


def _load_dotenv() -> None:
    # Load a project-local .env if present, without overriding already-exported variables.
    load_dotenv(override=False)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _list_env(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_embedding_model(provider: str) -> str:
    if provider == "qwen":
        return "text-embedding-v4"
    return "text-embedding-3-small"


@dataclass(slots=True)
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
    embedding_max_retries: int
    embedding_retry_backoff_ms: int
    openai_api_key: str | None
    openai_model: str
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
    top_k: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv()
        llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", llm_provider).strip().lower()
        return cls(
            llm_provider=llm_provider,
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            llm_api_style=os.getenv("LLM_API_STYLE", "responses").strip().lower(),
            embedding_provider=embedding_provider,
            embedding_api_key=os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL"),
            embedding_model=os.getenv("EMBEDDING_MODEL") or _default_embedding_model(embedding_provider),
            embedding_dimensions=_optional_int_env("EMBEDDING_DIMENSIONS"),
            embedding_max_retries=_int_env("EMBEDDING_MAX_RETRIES", 2),
            embedding_retry_backoff_ms=_int_env("EMBEDDING_RETRY_BACKOFF_MS", 800),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
            docs_exclude_patterns=_list_env("DOCS_EXCLUDE_PATTERNS"),
            index_path=Path(os.getenv("INDEX_PATH", "data/index/chunks.jsonl")),
            ingest_manifest_path=Path(os.getenv("INGEST_MANIFEST_PATH", "data/index/ingest_manifest.json")),
            eval_path=Path(os.getenv("EVAL_PATH", "data/evals/sample_eval.jsonl")),
            agent_runtime=os.getenv("AGENT_RUNTIME", "basic").strip().lower(),
            agents_max_turns=_int_env("AGENTS_MAX_TURNS", 6),
            vector_backend=os.getenv("VECTOR_BACKEND", "local").strip().lower(),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "local-docs-rag"),
            qdrant_timeout_s=_int_env("QDRANT_TIMEOUT_S", 30),
            top_k=_int_env("TOP_K", 4),
            chunk_strategy=os.getenv("CHUNK_STRATEGY", "markdown").strip().lower(),
            chunk_size=_int_env("CHUNK_SIZE", 800),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 120),
        )

    def with_runtime(self, runtime: str | None) -> "AppConfig":
        if runtime is None:
            return self
        return self.with_overrides(agent_runtime=runtime.strip().lower())

    def with_overrides(self, **overrides: object) -> "AppConfig":
        return replace(self, **overrides)
