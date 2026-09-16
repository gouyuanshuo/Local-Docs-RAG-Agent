"""Capabilities the agent runtimes expose as callable tools.

Each function takes plain arguments and returns domain objects or plain data so
it can be wrapped by the Agents SDK or called directly, and so the same
capability is testable without an agent in the loop. Search goes through the
retrieval pipeline and keeps embedding/reranker status.
"""

from __future__ import annotations

import datetime

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag
from local_docs_rag_agent.core import models


def list_documents(config: app_config.AppConfig) -> list[str]:
    """Return the POSIX paths of every document the agent can currently read."""
    return [
        path.as_posix()
        for path in rag.collect_document_paths(
            config.docs_dir, config.docs_exclude_patterns
        )
    ]


def search_documents(
    config: app_config.AppConfig, query: str, top_k: int | None = None
) -> models.RetrievalOutcome:
    """Search the index through the full retrieval pipeline.

    Args:
      config: The settings to search under.
      query: The question to search for.
      top_k: How many hits to return, or None for `config.top_k`.

    Returns:
      Hits plus embedding and reranker status. Going through
      :func:`rag.retrieve` keeps the tool's ordering and
      degradation identical to the answer path.
    """
    return rag.retrieve(config, query, top_k or config.top_k)


def get_system_time() -> str:
    """Return the current UTC time, so answers can reason about "today"."""
    return datetime.datetime.now(datetime.UTC).isoformat()
