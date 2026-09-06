from __future__ import annotations

import types
from typing import Any

import pytest
import qdrant_client

from local_docs_rag_agent import exceptions, models
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.rag import qdrant_store


class StubEmbeddingProvider(provider_base.EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    @property
    def status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="stub", mode="live")


class FallbackEmbeddingProvider(StubEmbeddingProvider):
    @property
    def status(self) -> models.ProviderStatus:
        return models.ProviderStatus(
            provider="stub", mode="fallback", reason="offline"
        )


def test_qdrant_client_receives_proxy_trust_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)

    qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key="test-key",
        collection_name="test-collection",
        timeout_s=12,
        embedding_provider=StubEmbeddingProvider(),
        trust_env=False,
    )

    assert captured["trust_env"] is False
    assert captured["timeout"] == 12


def test_winerror_10054_is_normalized_as_unreachable() -> None:
    message = (
        "[WinError 10054] An existing connection was forcibly closed "
        "by the remote host"
    )

    assert qdrant_store.looks_like_qdrant_unreachable(message)

    error = qdrant_store.qdrant_operation_error(
        operation="collection_exists",
        url="https://qdrant.example",
        collection_name="test-collection",
        exc=RuntimeError(message),
        trust_env=True,
    )

    assert "service appears unreachable" in str(error)
    assert "EXTERNAL_HTTP_TRUST_ENV=false" in str(error)


def test_unreachable_error_knows_proxy_is_already_disabled() -> None:
    error = qdrant_store.qdrant_operation_error(
        operation="collection_exists",
        url="https://qdrant.example",
        collection_name="test-collection",
        exc=RuntimeError("[WinError 10054] connection reset"),
        trust_env=False,
    )

    assert "already disabled" in str(error)


def test_qdrant_search_rejects_fallback_embedding_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            self.query_called = False

        def query_points(self, **kwargs: Any) -> None:
            self.query_called = True

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="test",
        timeout_s=10,
        embedding_provider=FallbackEmbeddingProvider(),
    )

    with pytest.raises(
        exceptions.ProviderUnavailableError, match="live embedding"
    ):
        store.search("question", top_k=3)

    assert store._client.query_called is False


def test_qdrant_save_rejects_chunk_without_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qdrant_client, "QdrantClient", lambda **kwargs: object()
    )
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="test",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )
    chunk = models.DocumentChunk(
        chunk_id="one",
        source_path="doc.md",
        title="Doc",
        text="content",
        chunk_index=0,
        start_char=0,
        end_char=7,
    )

    with pytest.raises(exceptions.VectorStoreError) as error:
        store.save([chunk])

    assert error.value.reason_code == "invalid_vectors"


def test_qdrant_load_paginates_until_offset_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {
            "chunk_id": chunk_id,
            "source_path": "doc.md",
            "title": "Doc",
            "text": chunk_id,
            "chunk_index": index,
            "start_char": index,
            "end_char": index + 1,
            "metadata": {},
        }
        for index, chunk_id in enumerate(["one", "two"])
    ]

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            self.offsets: list[object] = []

        def scroll(
            self, **kwargs: Any
        ) -> tuple[list[types.SimpleNamespace], object]:
            offset = kwargs.get("offset")
            self.offsets.append(offset)
            if offset is None:
                return [types.SimpleNamespace(payload=payloads[0])], "page-2"
            return [types.SimpleNamespace(payload=payloads[1])], None

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="test",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    chunks = store.load()

    assert [chunk.chunk_id for chunk in chunks] == ["one", "two"]
    assert store._client.offsets == [None, "page-2"]
