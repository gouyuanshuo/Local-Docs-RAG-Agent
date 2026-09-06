"""Builds the retrieval index from the command line."""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag


def run_ingest(config: app_config.AppConfig) -> None:
    """Ingest the configured documents and report the chunk count.

    Args:
      config: The settings to ingest under.
    """
    chunks = rag.ingest_documents(config)
    print(f"Ingested {len(chunks)} chunks into {config.vector_backend} store.")
