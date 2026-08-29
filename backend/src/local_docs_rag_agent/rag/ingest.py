from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ConfigurationError, ProviderUnavailableError
from local_docs_rag_agent.models import DocumentChunk
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.providers.factory import build_embedding_provider
from local_docs_rag_agent.rag.chunker import chunk_text
from local_docs_rag_agent.rag.manifest import IngestManifest
from local_docs_rag_agent.rag.store import ChunkStore, LocalJsonlChunkStore, QdrantChunkStore

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})


@dataclass(frozen=True, slots=True)
class IngestPlan:
    removed_sources: tuple[str, ...]
    changed_sources: tuple[str, ...]
    sources_to_index: tuple[str, ...]


def build_store(
    config: AppConfig,
    embedding_provider: EmbeddingProvider | None = None,
) -> ChunkStore:
    provider = embedding_provider or build_embedding_provider(config)
    if config.vector_backend == "qdrant":
        if not config.qdrant_url:
            raise ConfigurationError("QDRANT_URL must be set when VECTOR_BACKEND=qdrant")
        return QdrantChunkStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
            timeout_s=config.qdrant_timeout_s,
            embedding_provider=provider,
            trust_env=config.external_http_trust_env,
        )
    return LocalJsonlChunkStore(config.index_path, embedding_provider=provider)


def collect_document_paths(
    docs_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    _require_docs_directory(docs_dir)
    patterns = exclude_patterns or []
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_excluded(path, docs_dir, patterns)
    )


def ingest_documents(config: AppConfig) -> list[DocumentChunk]:
    source_texts = _read_source_texts(config)
    source_checksums = {source_path: _checksum(text) for source_path, text in source_texts.items()}
    previous_manifest = IngestManifest.load(config.ingest_manifest_path)
    embedding_provider = build_embedding_provider(config)
    store = build_store(config, embedding_provider=embedding_provider)
    desired_fingerprint = _index_fingerprint(
        config,
        embedding_mode="live" if config.embedding_api_key else "fallback",
    )
    plan = _build_ingest_plan(
        source_checksums=source_checksums,
        previous_manifest=previous_manifest,
        index_all=(
            config.vector_backend == "local"
            or (isinstance(store, QdrantChunkStore) and not store.collection_exists())
            or previous_manifest.index_fingerprint != desired_fingerprint
        ),
    )

    chunks = _chunk_sources(config, source_texts, plan.sources_to_index)
    _attach_embeddings(chunks, embedding_provider)
    if config.vector_backend == "qdrant" and chunks:
        _require_live_embeddings(embedding_provider)

    is_qdrant = config.vector_backend == "qdrant"
    store.save(
        chunks,
        removed_source_paths=list(plan.removed_sources) if is_qdrant else None,
        # Include changed-to-empty sources so their stale Qdrant points are deleted.
        replaced_source_paths=list(plan.sources_to_index) if is_qdrant else None,
    )
    previous_manifest.updated(
        source_checksums=source_checksums,
        indexed_sources=set(plan.sources_to_index),
        chunks=chunks,
        index_fingerprint=_index_fingerprint(
            config,
            embedding_mode=(
                embedding_provider.status.mode
                if chunks
                else ("live" if config.embedding_api_key else "fallback")
            ),
        ),
    ).save(config.ingest_manifest_path)
    return chunks


def ensure_index(config: AppConfig) -> None:
    manifest = IngestManifest.load(config.ingest_manifest_path)
    expected_fingerprint = _index_fingerprint(
        config,
        embedding_mode="live" if config.embedding_api_key else "fallback",
    )
    configuration_changed = manifest.index_fingerprint != expected_fingerprint
    if config.vector_backend == "qdrant":
        if configuration_changed or _is_qdrant_collection_missing(config):
            ingest_documents(config)
        return
    if configuration_changed or not config.index_path.exists():
        ingest_documents(config)


def _read_source_texts(config: AppConfig) -> dict[str, str]:
    source_texts: dict[str, str] = {}
    for path in collect_document_paths(
        config.docs_dir,
        config.docs_exclude_patterns,
    ):
        try:
            source_texts[path.as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError(f"Could not read source document {path}: {exc}") from exc
    return source_texts


def _build_ingest_plan(
    *,
    source_checksums: dict[str, str],
    previous_manifest: IngestManifest,
    index_all: bool,
) -> IngestPlan:
    previous_sources = previous_manifest.sources
    removed_sources = tuple(sorted(set(previous_sources) - set(source_checksums)))
    changed_sources = tuple(
        sorted(
            source_path
            for source_path, checksum in source_checksums.items()
            if source_path not in previous_sources
            or previous_sources[source_path].checksum != checksum
        )
    )
    sources_to_index = tuple(sorted(source_checksums)) if index_all else changed_sources
    return IngestPlan(
        removed_sources=removed_sources,
        changed_sources=changed_sources,
        sources_to_index=sources_to_index,
    )


def _chunk_sources(
    config: AppConfig,
    source_texts: dict[str, str],
    source_paths: tuple[str, ...],
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source_path in source_paths:
        chunks.extend(
            chunk_text(
                source_path=Path(source_path),
                text=source_texts[source_path],
                chunk_strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )
    return chunks


def _attach_embeddings(
    chunks: list[DocumentChunk],
    embedding_provider: EmbeddingProvider,
) -> None:
    if not chunks:
        return
    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ProviderUnavailableError(
            "Embedding provider returned a different number of vectors than input chunks",
            action_hint=f"Expected {len(chunks)} vectors, received {len(embeddings)}.",
        )
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if not embedding:
            raise ProviderUnavailableError(
                f"Embedding provider returned an empty vector for {chunk.chunk_id}"
            )
        chunk.embedding = embedding


def _require_live_embeddings(embedding_provider: EmbeddingProvider) -> None:
    status = embedding_provider.status
    if status.mode == "live":
        return
    reason = status.reason or "unknown_reason"
    action_hint = (
        "Try ingest again or increase EMBEDDING_MAX_RETRIES and EMBEDDING_RETRY_BACKOFF_MS."
        if reason.startswith("provider_transient_error:")
        else "Verify EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and EMBEDDING_MODEL."
    )
    raise ProviderUnavailableError(
        "Qdrant ingest requires a live embedding provider",
        action_hint=(
            f"Provider {status.provider!r} is in {status.mode!r} mode ({reason}). {action_hint}"
        ),
    )


def _require_docs_directory(docs_dir: Path) -> None:
    if not docs_dir.exists():
        raise ConfigurationError(
            f"DOCS_DIR does not exist: {docs_dir}",
            action_hint="Create the directory or set DOCS_DIR to an existing directory.",
        )
    if not docs_dir.is_dir():
        raise ConfigurationError(f"DOCS_DIR is not a directory: {docs_dir}")


def _is_excluded(path: Path, docs_dir: Path, exclude_patterns: list[str]) -> bool:
    relative_path = path.relative_to(docs_dir).as_posix()
    full_path = path.as_posix()
    return any(
        fnmatch(relative_path, pattern) or fnmatch(full_path, pattern)
        for pattern in exclude_patterns
    )


def _is_qdrant_collection_missing(config: AppConfig) -> bool:
    store = build_store(config)
    return isinstance(store, QdrantChunkStore) and not store.collection_exists()


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_fingerprint(config: AppConfig, *, embedding_mode: str) -> str:
    payload = {
        "vector_backend": config.vector_backend,
        "embedding_provider": config.embedding_provider,
        "embedding_base_url": config.embedding_base_url,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "embedding_mode": embedding_mode,
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
