from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from local_docs_rag_agent.exceptions import ProviderUnavailableError, VectorStoreError
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus, RetrievalHit
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.rag.scoring import build_retrieval_hit


class QdrantChunkStore:
    """Qdrant-backed chunk storage with normalized operational failures."""

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
            raise VectorStoreError(
                "qdrant-client is not installed",
                reason_code="dependency_missing",
                action_hint="Install the qdrant dependency with `pip install -e .[qdrant]`.",
            ) from exc

        self._client: Any = QdrantClient(
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
            raise self._operation_error("collection_exists", exc) from exc

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        vector_size = _validate_chunk_embeddings(chunks)
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is not installed",
                reason_code="dependency_missing",
                action_hint="Install the qdrant dependency with `pip install -e .[qdrant]`.",
            ) from exc

        try:
            collection_exists = bool(self._client.collection_exists(self._collection_name))
            if not collection_exists and vector_size is None:
                return
            if collection_exists and vector_size is not None:
                existing_size = _collection_vector_size(
                    self._client,
                    self._collection_name,
                )
                if existing_size != vector_size:
                    raise VectorStoreError(
                        "Qdrant collection vector size does not match incoming embeddings",
                        reason_code="vector_size_mismatch",
                        action_hint=(
                            f"Collection dimension is {existing_size}; incoming dimension is "
                            f"{vector_size}. Recreate the collection or use the matching model."
                        ),
                    )
            elif not collection_exists:
                assert vector_size is not None
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            self._ensure_source_path_index()
            self._delete_by_source_paths(removed_source_paths or [])
            self._delete_by_source_paths(replaced_source_paths or [])
            if chunks:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=[
                        models.PointStruct(
                            id=str(_qdrant_point_id(chunk.chunk_id)),
                            vector=_embedding_for_chunk(chunk),
                            payload=_chunk_payload(chunk),
                        )
                        for chunk in chunks
                    ],
                )
        except Exception as exc:
            raise self._operation_error("save", exc) from exc

    def load(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        next_offset: Any = None
        try:
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=1_000,
                    offset=next_offset,
                )
                chunks.extend(
                    DocumentChunk.from_dict(record.payload)
                    for record in records
                    if isinstance(record.payload, dict)
                )
                if next_offset is None:
                    return chunks
        except Exception as exc:
            raise self._operation_error("load", exc) from exc

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        query_vectors = self._embedding_provider.embed_texts([query])
        if self._embedding_provider.status.mode != "live":
            status = self._embedding_provider.status
            raise ProviderUnavailableError(
                "Qdrant search requires a live embedding provider",
                action_hint=(
                    f"Provider {status.provider!r} is in {status.mode!r} mode: "
                    f"{status.reason or 'no reason supplied'}. Check embedding configuration."
                ),
            )
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise ProviderUnavailableError(
                "Embedding provider did not return one non-empty query vector"
            )

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vectors[0],
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            points: Iterable[Any] = getattr(response, "points", response)
            return [
                build_retrieval_hit(
                    DocumentChunk.from_dict(point.payload),
                    float(point.score),
                )
                for point in points
                if isinstance(point.payload, dict)
            ]
        except Exception as exc:
            raise self._operation_error("search", exc) from exc

    @property
    def embedding_status(self) -> ProviderStatus:
        return self._embedding_provider.status

    def _delete_by_source_paths(self, source_paths: list[str]) -> None:
        if not source_paths:
            return
        from qdrant_client import models

        for source_path in sorted(set(source_paths)):
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

    def _ensure_source_path_index(self) -> None:
        from qdrant_client import models

        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="source_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            if "already exists" not in lowered and "already exist" not in lowered:
                raise

    def _operation_error(self, operation: str, exc: Exception) -> VectorStoreError:
        return _qdrant_operation_error(
            operation=operation,
            url=self._url,
            collection_name=self._collection_name,
            exc=exc,
            trust_env=self._trust_env,
        )


def _validate_chunk_embeddings(chunks: list[DocumentChunk]) -> int | None:
    if not chunks:
        return None
    missing = [chunk.chunk_id for chunk in chunks if not chunk.embedding]
    if missing:
        preview = ", ".join(missing[:3])
        raise VectorStoreError(
            f"Cannot save chunks without embeddings: {preview}",
            reason_code="invalid_vectors",
        )
    dimensions = {len(chunk.embedding or []) for chunk in chunks}
    if len(dimensions) != 1:
        raise VectorStoreError(
            "Cannot save embeddings with inconsistent dimensions",
            reason_code="invalid_vectors",
        )
    return dimensions.pop()


def _embedding_for_chunk(chunk: DocumentChunk) -> list[float]:
    if not chunk.embedding:
        raise VectorStoreError(
            f"Chunk {chunk.chunk_id} does not have an embedding",
            reason_code="invalid_vectors",
        )
    return chunk.embedding


def _chunk_payload(chunk: DocumentChunk) -> dict[str, object]:
    payload = chunk.to_dict()
    payload.pop("embedding", None)
    return payload


def _qdrant_point_id(chunk_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, chunk_id)


def _collection_vector_size(client: Any, collection_name: str) -> int:
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict):
        raise VectorStoreError(
            "Configured Qdrant collection uses named vectors, which are not supported",
            reason_code="invalid_collection",
            action_hint="Use a collection with one unnamed dense vector.",
        )
    raise VectorStoreError(
        "Unable to determine Qdrant collection vector size",
        reason_code="invalid_collection",
    )


def _qdrant_operation_error(
    operation: str,
    url: str,
    collection_name: str,
    exc: Exception,
    trust_env: bool | None = None,
) -> VectorStoreError:
    if isinstance(exc, VectorStoreError):
        return exc

    context = f"operation={operation} collection={collection_name} url={url}"
    detail = f"{exc.__class__.__name__}: {exc}"
    if _looks_like_qdrant_unreachable(str(exc)):
        proxy_hint = (
            "Environment proxies are already disabled for this client."
            if trust_env is False
            else "If stale proxy variables are present, set EXTERNAL_HTTP_TRUST_ENV=false."
        )
        return VectorStoreError(
            f"Qdrant service appears unreachable. {context}. error={detail}",
            reason_code="unreachable",
            action_hint=(
                "Check QDRANT_URL, network access, and whether the service is running. "
                f"{proxy_hint}"
            ),
        )
    return VectorStoreError(
        f"Qdrant operation failed. {context}. error={detail}",
        reason_code="operation_failed",
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
