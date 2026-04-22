from __future__ import annotations

import json
import math
import re
from uuid import uuid5, NAMESPACE_URL
from pathlib import Path
from typing import Protocol

from local_docs_rag_agent.models import CitationSpan, DocumentChunk, ProviderStatus, RetrievalHit
from local_docs_rag_agent.rag.embeddings import EmbeddingProvider


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class ChunkStore(Protocol):
    def save(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def load(self) -> list[DocumentChunk]:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    @property
    def embedding_status(self) -> ProviderStatus:
        raise NotImplementedError


class LocalJsonlChunkStore:
    def __init__(self, index_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self._index_path = index_path
        self._embedding_provider = embedding_provider

    def save(self, chunks: list[DocumentChunk]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=True) + "\n")

    def load(self) -> list[DocumentChunk]:
        if not self._index_path.exists():
            return []
        chunks: list[DocumentChunk] = []
        with self._index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunks.append(DocumentChunk.from_dict(json.loads(line)))
        return chunks

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_embedding = self._embedding_provider.embed_texts([query])[0]
        query_terms = _tokenize(query)
        hits: list[RetrievalHit] = []
        for chunk in self.load():
            dense_score = _cosine_similarity(query_embedding, chunk.embedding or [])
            lexical_score = _lexical_overlap_score(query_terms, _tokenize(chunk.text))
            score = max(dense_score, lexical_score, (dense_score + lexical_score) / 2)
            if score > 0:
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        score=score,
                        citation_span=_build_citation_span(chunk),
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    @property
    def embedding_status(self) -> ProviderStatus:
        return self._embedding_provider.status


class QdrantChunkStore:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider

    def save(self, chunks: list[DocumentChunk]) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc

        vectors = [chunk.embedding for chunk in chunks if chunk.embedding]
        if not vectors:
            raise ValueError("Embeddings must be present before saving to Qdrant.")

        vector_size = len(vectors[0])
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(_qdrant_point_id(chunk.chunk_id)),
                    vector=chunk.embedding,
                    payload=chunk.to_dict(),
                )
                for chunk in chunks
                if chunk.embedding
            ],
        )

    def load(self) -> list[DocumentChunk]:
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            with_payload=True,
            limit=10_000,
        )
        return [DocumentChunk.from_dict(record.payload) for record in records if record.payload]

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_embedding = self._embedding_provider.embed_texts([query])[0]
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)

        hits: list[RetrievalHit] = []
        for point in points:
            if not point.payload:
                continue
            chunk = DocumentChunk.from_dict(point.payload)
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(point.score),
                    citation_span=_build_citation_span(chunk),
                )
            )
        return hits

    @property
    def embedding_status(self) -> ProviderStatus:
        return self._embedding_provider.status


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _lexical_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / max(len(left), 1)


def _build_citation_span(chunk: DocumentChunk) -> CitationSpan:
    return CitationSpan(
        source_path=chunk.source_path,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        text=chunk.text,
    )


def _qdrant_point_id(chunk_id: str):
    return uuid5(NAMESPACE_URL, chunk_id)
