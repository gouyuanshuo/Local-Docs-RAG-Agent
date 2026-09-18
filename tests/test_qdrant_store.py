from __future__ import annotations

import traceback
import types
from typing import Any

import pytest
import qdrant_client

from local_docs_rag_agent.core import exceptions, models
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


def test_qdrant_store_reports_incremental_live_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.collection_names: list[str] = []

        def collection_exists(self, collection_name: str) -> bool:
            self.collection_names.append(collection_name)
            return True

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="test",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    assert store.exists() is True
    assert store.collection_exists() is True
    assert store.supports_incremental_updates is True
    assert store.requires_live_embeddings is True
    assert store._client.collection_names == ["test", "test"]


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


@pytest.mark.parametrize(
    ("raw_message", "reason_code"),
    [
        (
            "connection reset while using api-key-secret at path-secret",
            "unreachable",
        ),
        (
            "protocol exploded with query-secret and pass-secret",
            "operation_failed",
        ),
    ],
)
def test_qdrant_failure_traceback_excludes_url_and_exception_secrets(
    monkeypatch: pytest.MonkeyPatch,
    raw_message: str,
    reason_code: str,
) -> None:
    configured_url = (
        "https://user-secret:pass-secret@qdrant.example/path-secret"
        "?token=query-secret#fragment-secret"
    )

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def collection_exists(self, collection_name: str) -> bool:
            del collection_name
            raise RuntimeError(f"{raw_message}; url={configured_url}")

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url=configured_url,
        api_key="api-key-secret",
        collection_name="safe-collection",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    with pytest.raises(exceptions.VectorStoreError) as exc_info:
        store.collection_exists()

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.reason_code == reason_code
    assert "collection_exists" in str(exc_info.value)
    assert "safe-collection" in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)
    for secret in (
        "user-secret",
        "pass-secret",
        "path-secret",
        "query-secret",
        "fragment-secret",
        "api-key-secret",
        configured_url,
        raw_message,
    ):
        assert secret not in formatted


def test_qdrant_normalizer_redacts_pre_normalized_exception_text() -> None:
    raw = exceptions.VectorStoreError(
        "raw-domain-secret at https://user:password@example.invalid/private",
        reason_code="invalid_collection",
        action_hint="query-secret",
    )

    error = qdrant_store.qdrant_operation_error(
        operation="save",
        url="https://url-secret@example.invalid/path-secret",
        collection_name="safe-collection",
        exc=raw,
    )

    assert error.reason_code == "invalid_collection"
    assert "save" in str(error)
    assert "safe-collection" in str(error)
    assert "VectorStoreError" in str(error)
    for secret in (
        "raw-domain-secret",
        "user",
        "password",
        "private",
        "query-secret",
        "url-secret",
        "path-secret",
    ):
        assert secret not in str(error)


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


def test_qdrant_delete_collection_normalizes_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def delete_collection(self, collection_name: str) -> None:
            del collection_name
            raise RuntimeError("forced delete failure")

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="run-owned",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    with pytest.raises(exceptions.VectorStoreError) as exc_info:
        store.delete_collection()

    assert exc_info.value.reason_code == "operation_failed"
    assert "delete_collection" in str(exc_info.value)


def test_qdrant_delete_collection_rejects_false_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def delete_collection(self, collection_name: str) -> bool:
            del collection_name
            return False

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="run-owned",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    with pytest.raises(exceptions.VectorStoreError) as exc_info:
        store.delete_collection()

    assert exc_info.value.reason_code == "operation_failed"
    assert "delete_collection" in str(exc_info.value)


def test_qdrant_delete_collection_accepts_true_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def delete_collection(self, collection_name: str) -> bool:
            del collection_name
            return True

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)
    store = qdrant_store.QdrantChunkStore(
        url="https://qdrant.example",
        api_key=None,
        collection_name="run-owned",
        timeout_s=10,
        embedding_provider=StubEmbeddingProvider(),
    )

    store.delete_collection()


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
