"""Capabilities the agent runtimes expose as callable tools.

Each function takes plain arguments and returns plain data so it can be wrapped by the
Agents SDK or called directly by the basic runtime, and so the same capability is
testable without an agent in the loop.
"""

from __future__ import annotations

from datetime import UTC, datetime

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.rag.discovery import collect_document_paths
from local_docs_rag_agent.rag.store_factory import build_store


def list_documents(config: AppConfig) -> list[str]:
    """Return the POSIX paths of every document the agent can currently read."""

    return [
        path.as_posix()
        for path in collect_document_paths(config.docs_dir, config.docs_exclude_patterns)
    ]


def search_documents(
    config: AppConfig, query: str, top_k: int | None = None
) -> list[dict[str, str | float]]:
    """Search the index and return hits as plain data, with their source spans."""

    store = build_store(config)
    hits = store.search(query=query, top_k=top_k or config.top_k)
    return [
        {
            "source_path": hit.chunk.source_path,
            "title": hit.chunk.title,
            "score": round(hit.score, 4),
            "start_char": hit.citation_span.start_char,
            "end_char": hit.citation_span.end_char,
            "text": hit.chunk.text,
        }
        for hit in hits
    ]


def get_system_time() -> str:
    """Return the current UTC time, so answers can reason about "today"."""

    return datetime.now(UTC).isoformat()
