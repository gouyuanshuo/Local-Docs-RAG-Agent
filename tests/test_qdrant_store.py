from __future__ import annotations

from typing import Any

import qdrant_client

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.rag.store import (
    QdrantChunkStore,
    _looks_like_qdrant_unreachable,
    _qdrant_operation_error,
)


class StubEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(provider="stub", mode="live")


def test_qdrant_client_receives_proxy_trust_setting(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)

    QdrantChunkStore(
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
    message = "[WinError 10054] An existing connection was forcibly closed by the remote host"

    assert _looks_like_qdrant_unreachable(message)

    error = _qdrant_operation_error(
        operation="collection_exists",
        url="https://qdrant.example",
        collection_name="test-collection",
        exc=RuntimeError(message),
        trust_env=True,
    )

    assert "service appears unreachable" in str(error)
    assert "EXTERNAL_HTTP_TRUST_ENV=false" in str(error)


def test_unreachable_error_knows_when_environment_proxy_is_already_disabled() -> None:
    error = _qdrant_operation_error(
        operation="collection_exists",
        url="https://qdrant.example",
        collection_name="test-collection",
        exc=RuntimeError("[WinError 10054] connection reset"),
        trust_env=False,
    )

    assert "already disabled" in str(error)
