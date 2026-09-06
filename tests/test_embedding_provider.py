from __future__ import annotations

import types
from typing import Any, cast

import openai

from local_docs_rag_agent.providers import embedding


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, **request: Any) -> types.SimpleNamespace:
        batch = list(request["input"])
        self.calls.append(batch)
        return types.SimpleNamespace(
            data=[
                types.SimpleNamespace(embedding=[float(len(text))])
                for text in batch
            ]
        )


class RetryOnceEmbeddingsEndpoint(FakeEmbeddingsEndpoint):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def create(self, **request: Any) -> types.SimpleNamespace:
        self.attempts += 1
        if self.attempts == 1:
            error_type = type("APITimeoutError", (Exception,), {})
            raise error_type("timed out")
        return super().create(**request)


class ShortEmbeddingsEndpoint:
    def create(self, **request: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[1.0])]
        )


def _install_fake_endpoint(
    provider: embedding.OpenAICompatibleEmbeddingProvider,
    endpoint: object,
) -> None:
    """Point the provider at a duck-typed stub instead of a real OpenAI client.

    The provider only ever reaches for `client.embeddings.create`, so a
    namespace carrying that one attribute is enough; the cast records that the
    substitution is deliberate rather than a typing accident.
    """

    provider._client = cast(
        openai.OpenAI, types.SimpleNamespace(embeddings=endpoint)
    )


def _provider(
    batch_size: int, max_retries: int = 0
) -> embedding.OpenAICompatibleEmbeddingProvider:
    return embedding.OpenAICompatibleEmbeddingProvider(
        api_key=None,
        base_url=None,
        model="test-embedding",
        dimensions=None,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_backoff_ms=0,
        provider_label="test",
    )


def test_embed_texts_batches_requests_and_preserves_order() -> None:
    provider = _provider(batch_size=2)
    endpoint = FakeEmbeddingsEndpoint()
    _install_fake_endpoint(provider, endpoint)

    result = provider.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

    assert endpoint.calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert provider.status.mode == "live"
    assert provider.status.reason is None


def test_embed_texts_reports_recovery_after_retry() -> None:
    provider = _provider(batch_size=2, max_retries=1)
    endpoint = RetryOnceEmbeddingsEndpoint()
    _install_fake_endpoint(provider, endpoint)

    result = provider.embed_texts(["a", "bb"])

    assert result == [[1.0], [2.0]]
    assert endpoint.attempts == 2
    assert provider.status.mode == "live"
    assert provider.status.reason == "recovered_after_retry:1"


def test_embed_texts_falls_back_when_provider_returns_wrong_count() -> None:
    provider = _provider(batch_size=2)
    _install_fake_endpoint(provider, ShortEmbeddingsEndpoint())

    result = provider.embed_texts(["first", "second"])

    assert len(result) == 2
    assert all(len(vector) == 128 for vector in result)
    assert provider.status.mode == "fallback"
    assert "embedding_count_mismatch" in (provider.status.reason or "")
