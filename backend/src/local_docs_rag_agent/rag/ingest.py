"""Incremental ingest pipeline: discover, plan, chunk, embed, and commit.

The step order is deliberate and is what makes a partial or repeated run safe:

1. read every source document, so an unreadable file aborts before anything is written
2. load the typed manifest recorded by the previous run
3. compare the retrieval/embedding fingerprint and work out removed and changed sources
4. chunk only the sources that actually need reindexing
5. attach embeddings and reject empty or mismatched vectors
6. commit to the configured store, deleting points for removed and replaced sources
7. atomically replace the manifest

The fingerprint covers every setting that changes what a stored vector *means* — the
backend, the embedding model and dimension, whether embeddings are live or the hash
fallback, and the chunking parameters. When any of those change, an incremental update
would silently mix incomparable vectors, so the whole index is rebuilt instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.exceptions import ProviderUnavailableError
from local_docs_rag_agent.models import DocumentChunk
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.providers.factory import build_embedding_provider
from local_docs_rag_agent.rag.chunker import chunk_text
from local_docs_rag_agent.rag.discovery import read_source_texts
from local_docs_rag_agent.rag.manifest import IngestManifest
from local_docs_rag_agent.rag.qdrant_store import QdrantChunkStore
from local_docs_rag_agent.rag.store_factory import build_store


@dataclass(frozen=True, slots=True)
class IngestPlan:
    """The work one ingest run has to do, decided before anything is written."""

    removed_sources: tuple[str, ...]
    changed_sources: tuple[str, ...]
    sources_to_index: tuple[str, ...]


def ingest_documents(config: AppConfig) -> list[DocumentChunk]:
    """Bring the configured index up to date and return the chunks written."""

    source_texts = read_source_texts(config.docs_dir, config.docs_exclude_patterns)
    source_checksums = {
        source_path: source_checksum(text) for source_path, text in source_texts.items()
    }
    previous_manifest = IngestManifest.load(config.ingest_manifest_path)
    embedding_provider = build_embedding_provider(config)
    store = build_store(config, embedding_provider=embedding_provider)
    desired_fingerprint = index_fingerprint(config, embedding_mode=_expected_embedding_mode(config))
    plan = _build_ingest_plan(
        source_checksums=source_checksums,
        previous_manifest=previous_manifest,
        index_all=(
            # The local backend rewrites its whole file, so a partial plan buys nothing.
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
        index_fingerprint=index_fingerprint(
            config,
            # With no chunks the provider was never exercised, so record the intent.
            embedding_mode=(
                embedding_provider.status.mode if chunks else _expected_embedding_mode(config)
            ),
        ),
    ).save(config.ingest_manifest_path)
    return chunks


def ensure_index(config: AppConfig) -> None:
    """Ingest only if the index is missing or was built under different settings."""

    manifest = IngestManifest.load(config.ingest_manifest_path)
    configuration_changed = manifest.index_fingerprint != index_fingerprint(
        config,
        embedding_mode=_expected_embedding_mode(config),
    )
    if config.vector_backend == "qdrant":
        if configuration_changed or _is_qdrant_collection_missing(config):
            ingest_documents(config)
        return
    if configuration_changed or not config.index_path.exists():
        ingest_documents(config)


def source_checksum(text: str) -> str:
    """Return the content hash used to detect a changed source document."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_fingerprint(config: AppConfig, *, embedding_mode: str) -> str:
    """Return a hash of every setting that changes what a stored vector means."""

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


def _expected_embedding_mode(config: AppConfig) -> str:
    return "live" if config.embedding_api_key else "fallback"


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
    # Hash-fallback vectors are not comparable with the live vectors already stored in
    # a Qdrant collection, so writing them would quietly corrupt retrieval quality.
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


def _is_qdrant_collection_missing(config: AppConfig) -> bool:
    store = build_store(config)
    return isinstance(store, QdrantChunkStore) and not store.collection_exists()
