from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.rag import ingest_documents


def run_ingest(config: AppConfig) -> None:
    chunks = ingest_documents(config)
    print(f"Ingested {len(chunks)} chunks into {config.vector_backend} store.")
