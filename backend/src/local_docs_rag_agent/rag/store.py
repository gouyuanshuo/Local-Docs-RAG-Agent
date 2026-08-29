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
    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
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

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
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
            metadata_score = _metadata_overlap_score(query_terms, chunk)
            score = _hybrid_score(dense_score, lexical_score, metadata_score)
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
        timeout_s: int,
        embedding_provider: EmbeddingProvider,
        trust_env: bool = True,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc

        self._client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_s,
            trust_env=trust_env,
        )
        self._url = url
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._trust_env = trust_env

    def collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self._collection_name))
        except Exception as exc:
            raise _qdrant_operation_error(
                operation="collection_exists",
                url=self._url,
                collection_name=self._collection_name,
                exc=exc,
                trust_env=self._trust_env,
            ) from exc

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc

        removed_source_paths = removed_source_paths or []
        replaced_source_paths = replaced_source_paths or []
        vectors = [chunk.embedding for chunk in chunks if chunk.embedding]
        vector_size = len(vectors[0]) if vectors else None

        try:
            collection_exists = self._client.collection_exists(self._collection_name)
            if not collection_exists and vector_size is None:
                return
            if collection_exists:
                if vector_size is not None:
                    existing_size = _collection_vector_size(self._client, self._collection_name)
                    if existing_size != vector_size:
                        raise RuntimeError(
                            "Qdrant collection vector size mismatch. "
                            f"collection={existing_size}, incoming={vector_size}. "
                            "Delete/recreate the collection or re-run ingest with a consistent embedding mode."
                        )
            else:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
            self._ensure_source_path_index()

            self._delete_by_source_paths(removed_source_paths)
            self._delete_by_source_paths(replaced_source_paths)

            if vectors:
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
        except Exception as exc:
            raise _qdrant_operation_error(
                operation="save",
                url=self._url,
                collection_name=self._collection_name,
                exc=exc,
                trust_env=self._trust_env,
            ) from exc

    def load(self) -> list[DocumentChunk]:
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=True,
                limit=10_000,
            )
            return [DocumentChunk.from_dict(record.payload) for record in records if record.payload]
        except Exception as exc:
            raise _qdrant_operation_error(
                operation="load",
                url=self._url,
                collection_name=self._collection_name,
                exc=exc,
                trust_env=self._trust_env,
            ) from exc

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        query_embedding = self._embedding_provider.embed_texts([query])[0]
        try:
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
        except Exception as exc:
            raise _qdrant_operation_error(
                operation="search",
                url=self._url,
                collection_name=self._collection_name,
                exc=exc,
                trust_env=self._trust_env,
            ) from exc

    @property
    def embedding_status(self) -> ProviderStatus:
        return self._embedding_provider.status

    def _delete_by_source_paths(self, source_paths: list[str]) -> None:
        if not source_paths:
            return
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc

        for source_path in source_paths:
            try:
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source_path",
                                    match=models.MatchValue(value=source_path),
                                )
                            ]
                        )
                    ),
                )
            except Exception as exc:
                raise _qdrant_operation_error(
                    operation="delete",
                    url=self._url,
                    collection_name=self._collection_name,
                    exc=exc,
                    trust_env=self._trust_env,
                ) from exc

    def _ensure_source_path_index(self) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Install with `pip install -e .[qdrant]`."
            ) from exc
        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="source_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" in message or "exists" in message:
                return
            raise


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


def _metadata_overlap_score(query_terms: set[str], chunk: DocumentChunk) -> float:
    metadata_terms = _tokenize(chunk.title)
    section_title = chunk.metadata.get("section_title")
    source_title = chunk.metadata.get("source_title")
    if isinstance(section_title, str):
        metadata_terms |= _tokenize(section_title)
    if isinstance(source_title, str):
        metadata_terms |= _tokenize(source_title)
    return _lexical_overlap_score(query_terms, metadata_terms)


def _hybrid_score(dense_score: float, lexical_score: float, metadata_score: float) -> float:
    weighted_mix = (dense_score * 0.65) + (lexical_score * 0.25) + (metadata_score * 0.10)
    return max(dense_score, lexical_score, weighted_mix)


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


def _collection_vector_size(client, collection_name: str) -> int:
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()))
        if hasattr(first, "size"):
            return int(first.size)
    raise RuntimeError("Unable to determine Qdrant collection vector size.")


def _qdrant_operation_error(
    operation: str,
    url: str,
    collection_name: str,
    exc: Exception,
    trust_env: bool | None = None,
) -> RuntimeError:
    exc_name = exc.__class__.__name__
    message = str(exc)
    if _looks_like_qdrant_unreachable(message):
        if trust_env is False:
            proxy_hint = "Environment proxies are already disabled for this client."
        else:
            proxy_hint = "If stale proxy variables are present, set EXTERNAL_HTTP_TRUST_ENV=false."
        return RuntimeError(
            "Qdrant operation failed because the service appears unreachable.\n"
            f"operation={operation} collection={collection_name} url={url}\n"
            f"error={exc_name}: {message}\n"
            "action_hint=check QDRANT_URL, confirm network access, and verify the service is running. "
            f"{proxy_hint}"
        )
    return RuntimeError(
        "Qdrant operation failed.\n"
        f"operation={operation} collection={collection_name} url={url}\n"
        f"error={exc_name}: {message}"
    )


def _looks_like_qdrant_unreachable(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "connection refused",
            "actively refused",
            "failed to establish a new connection",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "winerror 10061",
            "winerror 10054",
            "connecterror",
            "responsehandlingexception",
            "connection reset",
            "forcibly closed",
            "server disconnected",
        )
    )
