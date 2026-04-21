from __future__ import annotations

from datetime import datetime, timezone

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.rag.ingest import build_store, collect_document_paths


def list_documents(config: AppConfig) -> list[str]:
    return [path.as_posix() for path in collect_document_paths(config.docs_dir)]


def search_documents(config: AppConfig, query: str, top_k: int | None = None) -> list[dict[str, str | float]]:
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
    return datetime.now(timezone.utc).isoformat()
