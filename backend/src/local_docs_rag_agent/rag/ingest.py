from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import DocumentChunk
from local_docs_rag_agent.providers.factory import build_embedding_provider
from local_docs_rag_agent.rag.chunker import chunk_text
from local_docs_rag_agent.rag.store import ChunkStore, LocalJsonlChunkStore, QdrantChunkStore


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def build_store(config: AppConfig) -> ChunkStore:
    embedding_provider = build_embedding_provider(config)
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise ValueError("QDRANT_URL must be set when VECTOR_BACKEND=qdrant")
        return QdrantChunkStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            embedding_provider=embedding_provider,
        )
    return LocalJsonlChunkStore(config.index_path, embedding_provider=embedding_provider)


def collect_document_paths(docs_dir: Path, exclude_patterns: list[str] | None = None) -> list[Path]:
    patterns = exclude_patterns or []
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_excluded(path, docs_dir, patterns)
    )


def _is_excluded(path: Path, docs_dir: Path, exclude_patterns: list[str]) -> bool:
    relative_path = path.relative_to(docs_dir).as_posix()
    full_path = path.as_posix()
    return any(fnmatch(relative_path, pattern) or fnmatch(full_path, pattern) for pattern in exclude_patterns)


def ingest_documents(config: AppConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in collect_document_paths(config.docs_dir, config.docs_exclude_patterns):
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_text(
                source_path=path,
                text=text,
                chunk_strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )

    embedding_provider = build_embedding_provider(config)
    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    store = build_store(config)
    store.save(chunks)
    return chunks


def ensure_index(config: AppConfig) -> None:
    if config.vector_backend != "local":
        return
    if config.index_path.exists():
        return
    ingest_documents(config)
