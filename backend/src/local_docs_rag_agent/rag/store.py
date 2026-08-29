from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from local_docs_rag_agent.exceptions import DataFormatError, ProviderUnavailableError
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus, RetrievalHit
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.rag.file_io import atomic_write_text
from local_docs_rag_agent.rag.qdrant_store import (
    QdrantChunkStore,
    _looks_like_qdrant_unreachable,
    _qdrant_operation_error,
)
from local_docs_rag_agent.rag.scoring import build_retrieval_hit, score_local_chunk, tokenize


class ChunkStore(Protocol):
    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None: ...

    def load(self) -> list[DocumentChunk]: ...

    def search(self, query: str, top_k: int) -> list[RetrievalHit]: ...

    @property
    def embedding_status(self) -> ProviderStatus: ...


class LocalJsonlChunkStore:
    """Small, inspectable on-disk store for local development and fallback use."""

    def __init__(self, index_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self._index_path = index_path
        self._embedding_provider = embedding_provider

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        del removed_source_paths, replaced_source_paths
        content = "".join(f"{json.dumps(chunk.to_dict(), ensure_ascii=True)}\n" for chunk in chunks)
        atomic_write_text(self._index_path, content)

    def load(self) -> list[DocumentChunk]:
        if not self._index_path.exists():
            return []
        chunks: list[DocumentChunk] = []
        line_number: int | str = "unknown"
        try:
            with self._index_path.open("r", encoding="utf-8") as handle:
                for current_line_number, line in enumerate(handle, start=1):
                    line_number = current_line_number
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("chunk must be a JSON object")
                    chunks.append(DocumentChunk.from_dict(payload))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            DataFormatError,
            TypeError,
            ValueError,
        ) as exc:
            raise DataFormatError(
                f"Could not read local chunk index {self._index_path} at line {line_number}: {exc}"
            ) from exc
        return chunks

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        query_vectors = self._embedding_provider.embed_texts([query])
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise ProviderUnavailableError(
                "Embedding provider did not return one non-empty query vector"
            )
        query_embedding = query_vectors[0]
        query_terms = tokenize(query)
        hits = [
            build_retrieval_hit(
                chunk,
                score_local_chunk(
                    query_embedding=query_embedding,
                    query_terms=query_terms,
                    chunk=chunk,
                ),
            )
            for chunk in self.load()
        ]
        hits = [hit for hit in hits if hit.score > 0]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    @property
    def embedding_status(self) -> ProviderStatus:
        return self._embedding_provider.status


__all__ = [
    "ChunkStore",
    "LocalJsonlChunkStore",
    "QdrantChunkStore",
    "_looks_like_qdrant_unreachable",
    "_qdrant_operation_error",
]
