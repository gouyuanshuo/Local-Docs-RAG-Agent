"""Capabilities the agent runtimes expose as callable tools.

Each function takes plain arguments and returns plain data so it can be wrapped
by the Agents SDK or called directly by the basic runtime, and so the same
capability is testable without an agent in the loop.
"""

from __future__ import annotations

import datetime

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.rag import discovery, pipeline


def list_documents(config: app_config.AppConfig) -> list[str]:
    """Return the POSIX paths of every document the agent can currently read."""

    return [
        path.as_posix()
        for path in discovery.collect_document_paths(
            config.docs_dir, config.docs_exclude_patterns
        )
    ]


def search_documents(
    config: app_config.AppConfig, query: str, top_k: int | None = None
) -> list[dict[str, str | float]]:
    """Search the index and return hits as plain data, with their source spans.

    This goes through the full retrieval pipeline, reranking included, so the
    tool cannot report a different ordering from the one an answer would have
    been built on.
    """

    hits = pipeline.retrieve(config, query, top_k or config.top_k).hits
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

    return datetime.datetime.now(datetime.UTC).isoformat()
