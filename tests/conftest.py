"""Shared pytest configuration.

The suite is meant to be deterministic and offline, but `AppConfig.from_env`
reads the process environment and loads a project-local `.env`. Without
isolation a developer's own `.env` silently supplies API keys and endpoints, so
a test can pass on their machine and fail in CI, or pass for a reason that has
nothing to do with what it asserts.

The autouse fixture below neutralizes both sources for every test. A test that
needs a particular setting still sets it explicitly with `monkeypatch.setenv`,
which then reads as part of the test rather than as an accident of the machine.

The dotenv loader is patched where it is defined, in `env`, rather than on the
module that calls it. Under module-only imports there is no second binding to
patch, and patching the definition covers every caller rather than the one that
happened to be named here.
"""

from __future__ import annotations

import pytest

from local_docs_rag_agent import env

APPLICATION_ENVIRONMENT_VARIABLES = (
    "AGENTS_MAX_TURNS",
    "AGENT_RUNTIME",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "CHUNK_STRATEGY",
    "DOCS_DIR",
    "DOCS_EXCLUDE_PATTERNS",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MAX_RETRIES",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_RETRY_BACKOFF_MS",
    "EVAL_PATH",
    "EXTERNAL_HTTP_TRUST_ENV",
    "INDEX_PATH",
    "INGEST_MANIFEST_PATH",
    "LLM_API_KEY",
    "LLM_API_STYLE",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_TIMEOUT_S",
    "QDRANT_URL",
    "RERANKER",
    "RERANK_CANDIDATE_K",
    "RERANK_MODEL",
    "RETRIEVAL_CANDIDATE_K",
    "RETRIEVAL_STRATEGY",
    "RRF_K",
    "TOP_K",
    "VECTOR_BACKEND",
)


@pytest.fixture(autouse=True)
def isolated_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against defaults, not the developer's machine."""

    monkeypatch.setattr(env, "load_project_dotenv", lambda: None)
    for name in APPLICATION_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
