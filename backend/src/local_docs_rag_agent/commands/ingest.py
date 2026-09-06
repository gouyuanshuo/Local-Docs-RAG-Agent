from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag


def run_ingest(config: app_config.AppConfig) -> None:
    chunks = rag.ingest_documents(config)
    print(f"Ingested {len(chunks)} chunks into {config.vector_backend} store.")
