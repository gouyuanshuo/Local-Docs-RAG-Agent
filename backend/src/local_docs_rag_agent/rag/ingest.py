from __future__ import annotations

import hashlib
import json
from fnmatch import fnmatch
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import DocumentChunk
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.providers.factory import build_embedding_provider
from local_docs_rag_agent.rag.chunker import chunk_text
from local_docs_rag_agent.rag.store import ChunkStore, LocalJsonlChunkStore, QdrantChunkStore


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def build_store(config: AppConfig, embedding_provider: EmbeddingProvider | None = None) -> ChunkStore:
    embedding_provider = embedding_provider or build_embedding_provider(config)
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise ValueError("QDRANT_URL must be set when VECTOR_BACKEND=qdrant")
        return QdrantChunkStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            timeout_s=config.qdrant_timeout_s,
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
    source_texts = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in collect_document_paths(config.docs_dir, config.docs_exclude_patterns)
    }
    source_checksums = {source_path: _checksum(text) for source_path, text in source_texts.items()}

    embedding_provider = build_embedding_provider(config)
    previous_manifest = _read_manifest(config.ingest_manifest_path)
    previous_sources = previous_manifest.get("sources", {})

    removed_sources = sorted(set(previous_sources) - set(source_texts))
    changed_sources = sorted(
        source_path
        for source_path, checksum in source_checksums.items()
        if previous_sources.get(source_path, {}).get("checksum") != checksum
    )

    sources_to_chunk = sorted(source_texts) if config.vector_backend == "local" else changed_sources
    chunks: list[DocumentChunk] = []
    for source_path in sources_to_chunk:
        text = source_texts[source_path]
        chunks.extend(
            chunk_text(
                source_path=Path(source_path),
                text=text,
                chunk_strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )

    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks]) if chunks else []
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    if config.vector_backend == "qdrant" and chunks and embedding_provider.status.mode != "live":
        status = embedding_provider.status
        reason = status.reason or "unknown_reason"
        action_hint = (
            "embedding transient failure exhausted retries; try ingest again or increase "
            "EMBEDDING_MAX_RETRIES / EMBEDDING_RETRY_BACKOFF_MS."
            if reason.startswith("provider_transient_error:")
            else "provider is not live; verify EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL."
        )
        raise RuntimeError(
            "Qdrant ingest aborted because embedding provider is not live.\n"
            f"provider={status.provider} mode={status.mode} reason={reason}\n"
            f"action_hint={action_hint}"
        )

    store = build_store(config, embedding_provider=embedding_provider)
    replaced_sources = sorted({chunk.source_path for chunk in chunks}) if config.vector_backend == "qdrant" else []
    store.save(
        chunks,
        removed_source_paths=removed_sources if config.vector_backend == "qdrant" else None,
        replaced_source_paths=replaced_sources if config.vector_backend == "qdrant" else None,
    )
    _write_manifest(
        config.ingest_manifest_path,
        previous_sources=previous_sources,
        source_checksums=source_checksums,
        changed_chunks=chunks,
        removed_sources=removed_sources,
    )
    return chunks


def ensure_index(config: AppConfig) -> None:
    if config.vector_backend != "local":
        return
    if config.index_path.exists():
        return
    ingest_documents(config)


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"sources": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "sources" not in payload:
        return {"sources": {}}
    if not isinstance(payload["sources"], dict):
        return {"sources": {}}
    return payload


def _write_manifest(
    path: Path,
    previous_sources: dict[str, object],
    source_checksums: dict[str, str],
    changed_chunks: list[DocumentChunk],
    removed_sources: list[str],
) -> None:
    next_sources = {
        source_path: dict(payload) if isinstance(payload, dict) else {}
        for source_path, payload in previous_sources.items()
    }
    for source_path in removed_sources:
        next_sources.pop(source_path, None)

    chunk_ids_by_source: dict[str, list[str]] = {}
    for chunk in changed_chunks:
        chunk_ids_by_source.setdefault(chunk.source_path, []).append(chunk.chunk_id)

    for source_path, checksum in source_checksums.items():
        existing = next_sources.get(source_path, {})
        if not isinstance(existing, dict):
            existing = {}
        next_sources[source_path] = {
            "checksum": checksum,
            "chunk_ids": chunk_ids_by_source.get(source_path, existing.get("chunk_ids", [])),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": next_sources}, ensure_ascii=True, indent=2), encoding="utf-8")
