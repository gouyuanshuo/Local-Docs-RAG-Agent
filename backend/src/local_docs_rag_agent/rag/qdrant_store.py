"""Qdrant-backed chunk store with normalized operational failures.

`qdrant_client` is imported lazily so the package installs and runs without the
`qdrant` extra, and every client call is funnelled through
`qdrant_operation_error` so callers see a `VectorStoreError` with a stable
`reason_code` instead of a raw transport exception. The eval matrix relies on
those codes to tell "Qdrant is not reachable here" apart from "this
configuration genuinely failed".
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from local_docs_rag_agent import exceptions, models
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.rag import retrieval, scoring


class QdrantChunkStore:
    """Qdrant-backed chunk storage with normalized operational failures."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        timeout_s: int,
        embedding_provider: provider_base.EmbeddingProvider,
        trust_env: bool = True,
        settings: retrieval.RetrievalSettings | None = None,
    ) -> None:
        """Connect to a Qdrant collection.

        Args:
          url: Base URL of the Qdrant service.
          api_key: Credential, or None for an unsecured instance.
          collection_name: Collection holding this project's points.
          timeout_s: Per-request timeout.
          embedding_provider: Used to embed the query at search time.
          trust_env: Whether to honour environment proxy variables.
          settings: Ranking knobs. Defaults to the project defaults.

        Raises:
          VectorStoreError: With `dependency_missing` when the optional
            `qdrant-client` package is not installed.
        """
        try:
            import qdrant_client
        except ImportError as exc:
            raise exceptions.VectorStoreError(
                "qdrant-client is not installed",
                reason_code="dependency_missing",
                action_hint=(
                    "Install the qdrant dependency with "
                    "`pip install -e .[qdrant]`."
                ),
            ) from exc

        self._client: Any = qdrant_client.QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_s,
            trust_env=trust_env,
        )
        self._url = url
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._trust_env = trust_env
        self._settings = settings or retrieval.RetrievalSettings()

    def collection_exists(self) -> bool:
        """Report whether the configured collection exists.

        Returns:
          True if the collection is present.

        Raises:
          VectorStoreError: If the service cannot be reached or refuses
            the request.
        """
        try:
            return bool(self._client.collection_exists(self._collection_name))
        except Exception as exc:
            raise self._operation_error("collection_exists", exc) from exc

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        """Create the collection if needed, then apply this ingest's changes.

        Args:
          chunks: Chunks to upsert. Every one must carry an embedding.
          removed_source_paths: Documents deleted since the last ingest,
            whose points must go with them.
          replaced_source_paths: Documents re-chunked by this run, whose
            old points are deleted before the new ones are written.

        Raises:
          VectorStoreError: If a chunk lacks an embedding, the embedding
            dimensions disagree with the collection, or the service call
            fails.
        """
        vector_size = _validate_chunk_embeddings(chunks)
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise exceptions.VectorStoreError(
                "qdrant-client is not installed",
                reason_code="dependency_missing",
                action_hint=(
                    "Install the qdrant dependency with "
                    "`pip install -e .[qdrant]`."
                ),
            ) from exc

        try:
            collection_exists = bool(
                self._client.collection_exists(self._collection_name)
            )
            if not collection_exists and vector_size is None:
                return
            if collection_exists and vector_size is not None:
                existing_size = _collection_vector_size(
                    self._client,
                    self._collection_name,
                )
                if existing_size != vector_size:
                    raise exceptions.VectorStoreError(
                        "Qdrant collection vector size does not match "
                        "incoming embeddings",
                        reason_code="vector_size_mismatch",
                        action_hint=(
                            f"Collection dimension is {existing_size}; "
                            f"incoming dimension is {vector_size}. "
                            "Recreate the collection or use the "
                            "matching model."
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

    def load(self) -> list[models.DocumentChunk]:
        """Read every stored chunk, paging through the collection.

        Returns:
          The stored chunks, without their vectors: Qdrant returns
          payloads only.

        Raises:
          VectorStoreError: If the service call fails.
        """
        chunks: list[models.DocumentChunk] = []
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
                    models.DocumentChunk.from_dict(record.payload)
                    for record in records
                    if isinstance(record.payload, dict)
                )
                if next_offset is None:
                    return chunks
        except Exception as exc:
            raise self._operation_error("load", exc) from exc

    def search(self, query: str, top_k: int) -> list[models.RetrievalHit]:
        """Rank the collection against `query`.

        Args:
          query: The question to rank against.
          top_k: Maximum hits to return.

        Returns:
          At most `top_k` hits, best first. A fused strategy asks the
          server for a wider window and re-ranks it locally, because the
          server returns no vectors to fuse with.

        Raises:
          ProviderUnavailableError: If the embedding provider is not live.
            A fallback vector would query the collection with coordinates
            that mean nothing in it.
          VectorStoreError: If the service call fails.
        """
        if top_k <= 0:
            return []
        query_vectors = self._embedding_provider.embed_texts([query])
        if self._embedding_provider.status.mode != "live":
            status = self._embedding_provider.status
            raise exceptions.ProviderUnavailableError(
                "Qdrant search requires a live embedding provider",
                action_hint=(
                    f"Provider {status.provider!r} is in {status.mode!r} mode: "
                    f"{status.reason or 'no reason supplied'}. "
                    "Check embedding configuration."
                ),
            )
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise exceptions.ProviderUnavailableError(
                "Embedding provider did not return one non-empty query vector"
            )

        # The dense-only strategies take exactly what they need from the server;
        # the fused ones need a wider window to re-rank within.
        fuses_locally = retrieval.needs_candidate_window(
            self._settings.strategy
        )
        limit = self._settings.window(top_k) if fuses_locally else top_k
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vectors[0],
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points: Iterable[Any] = getattr(response, "points", response)
            dense_hits = [
                scoring.build_retrieval_hit(
                    models.DocumentChunk.from_dict(point.payload),
                    float(point.score),
                )
                for point in points
                if isinstance(point.payload, dict)
            ]
        except Exception as exc:
            raise self._operation_error("search", exc) from exc

        # `blended` needs the chunk vectors to combine signals, and the server
        # does not return them, so on this backend it means what it has always
        # meant here: the server's own dense ordering.
        if not fuses_locally:
            return dense_hits[:top_k]
        return retrieval.rerank_dense_hits(
            query=query,
            dense_hits=dense_hits,
            top_k=top_k,
            settings=self._settings,
        )

    @property
    def embedding_status(self) -> models.ProviderStatus:
        """Report the health of the embedding provider backing this store."""
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
            if (
                "already exists" not in lowered
                and "already exist" not in lowered
            ):
                raise

    def _operation_error(
        self, operation: str, exc: Exception
    ) -> exceptions.VectorStoreError:
        return qdrant_operation_error(
            operation=operation,
            url=self._url,
            collection_name=self._collection_name,
            exc=exc,
            trust_env=self._trust_env,
        )


def _validate_chunk_embeddings(
    chunks: list[models.DocumentChunk],
) -> int | None:
    if not chunks:
        return None
    missing = [chunk.chunk_id for chunk in chunks if not chunk.embedding]
    if missing:
        preview = ", ".join(missing[:3])
        raise exceptions.VectorStoreError(
            f"Cannot save chunks without embeddings: {preview}",
            reason_code="invalid_vectors",
        )
    dimensions = {len(chunk.embedding or []) for chunk in chunks}
    if len(dimensions) != 1:
        raise exceptions.VectorStoreError(
            "Cannot save embeddings with inconsistent dimensions",
            reason_code="invalid_vectors",
        )
    return dimensions.pop()


def _embedding_for_chunk(chunk: models.DocumentChunk) -> list[float]:
    if not chunk.embedding:
        raise exceptions.VectorStoreError(
            f"Chunk {chunk.chunk_id} does not have an embedding",
            reason_code="invalid_vectors",
        )
    return chunk.embedding


def _chunk_payload(chunk: models.DocumentChunk) -> dict[str, object]:
    payload = chunk.to_dict()
    payload.pop("embedding", None)
    return payload


def _qdrant_point_id(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)


def _collection_vector_size(client: Any, collection_name: str) -> int:
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict):
        raise exceptions.VectorStoreError(
            "Configured Qdrant collection uses named vectors, which are "
            "not supported",
            reason_code="invalid_collection",
            action_hint="Use a collection with one unnamed dense vector.",
        )
    raise exceptions.VectorStoreError(
        "Unable to determine Qdrant collection vector size",
        reason_code="invalid_collection",
    )


def qdrant_operation_error(
    operation: str,
    url: str,
    collection_name: str,
    exc: Exception,
    trust_env: bool | None = None,
) -> exceptions.VectorStoreError:
    """Translate a client failure into a `VectorStoreError` with a code.

    Args:
      operation: The client call that failed, named in the message.
      url: The service the call was made against.
      collection_name: The collection the call addressed.
      exc: The raw failure.
      trust_env: Whether the client honoured environment proxies,
        which decides which proxy hint is worth giving.

    Returns:
      The normalized error. An already-normalized one is passed
      through unchanged, so a specific diagnosis such as a
      vector-size mismatch is not flattened into a generic failure.
    """
    if isinstance(exc, exceptions.VectorStoreError):
        return exc

    context = f"operation={operation} collection={collection_name} url={url}"
    detail = f"{exc.__class__.__name__}: {exc}"
    if looks_like_qdrant_unreachable(str(exc)):
        proxy_hint = (
            "Environment proxies are already disabled for this client."
            if trust_env is False
            else "If stale proxy variables are present, set "
            "EXTERNAL_HTTP_TRUST_ENV=false."
        )
        return exceptions.VectorStoreError(
            f"Qdrant service appears unreachable. {context}. error={detail}",
            reason_code="unreachable",
            action_hint=(
                "Check QDRANT_URL, network access, and whether the "
                "service is running. "
                f"{proxy_hint}"
            ),
        )
    return exceptions.VectorStoreError(
        f"Qdrant operation failed. {context}. error={detail}",
        reason_code="operation_failed",
    )


def looks_like_qdrant_unreachable(message: str) -> bool:
    """Report whether an error message describes a connectivity failure.

    Matching on message text is deliberate: the client wraps transport
    errors from several libraries, so the exception type alone does not
    identify them.

    Args:
      message: The failure text to classify.

    Returns:
      True if the message describes a connectivity failure, which the
      eval matrix reports as a skip rather than as a bad result.
    """
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
