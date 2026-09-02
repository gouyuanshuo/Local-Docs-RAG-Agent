"""OpenAI-compatible embedding provider with batching, retry, and a hash fallback.

Requests are batched because hosted endpoints cap inputs per call, and transient
failures are retried with exponential backoff before the provider gives up. On giving
up it returns deterministic hash embeddings and reports `fallback`: those vectors keep
local retrieval working offline, but they are not comparable with live vectors, which
is why Qdrant ingest refuses them.
"""

from __future__ import annotations

import hashlib
import math
import time

from openai import OpenAI

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.providers.base import EmbeddingProvider
from local_docs_rag_agent.providers.errors import (
    is_transient_provider_error,
    provider_error_reason,
)
from local_docs_rag_agent.providers.openai_client import build_sync_openai_client


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str,
        dimensions: int | None,
        batch_size: int = 16,
        max_retries: int = 2,
        retry_backoff_ms: int = 800,
        provider_label: str = "embedding",
        trust_env: bool = True,
    ) -> None:
        self._provider_label = provider_label.lower()
        self._trust_env = trust_env
        self._status = ProviderStatus(provider=self._provider_label, mode="ready")
        self._client: OpenAI | None = None
        if api_key:
            self._client = build_sync_openai_client(
                api_key=api_key,
                base_url=base_url,
                trust_env=trust_env,
            )
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_ms = max(0, retry_backoff_ms)
        if not api_key:
            self._status = ProviderStatus(
                provider=self._provider_label, mode="fallback", reason="missing_api_key"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._client:
            return [_hash_embed(text) for text in texts]

        results: list[list[float]] = []
        total_attempts = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            attempt = 0
            while True:
                try:
                    if self._dimensions is None:
                        response = self._client.embeddings.create(
                            model=self._model,
                            input=batch,
                        )
                    else:
                        response = self._client.embeddings.create(
                            model=self._model,
                            input=batch,
                            dimensions=self._dimensions,
                        )
                    batch_embeddings = [item.embedding for item in response.data]
                    if len(batch_embeddings) != len(batch):
                        raise RuntimeError(
                            "embedding_count_mismatch:"
                            f"expected={len(batch)}:received={len(batch_embeddings)}"
                        )
                    total_attempts += attempt
                    results.extend(batch_embeddings)
                    break
                except Exception as exc:
                    if is_transient_provider_error(exc) and attempt < self._max_retries:
                        sleep_ms = self._retry_backoff_ms * (2**attempt)
                        if sleep_ms > 0:
                            time.sleep(sleep_ms / 1000.0)
                        attempt += 1
                        continue
                    self._status = ProviderStatus(
                        provider=self._provider_label,
                        mode="fallback",
                        reason=_fallback_reason(
                            exc=exc, attempts=attempt + 1, max_retries=self._max_retries
                        ),
                    )
                    return [_hash_embed(text) for text in texts]

        reason = f"recovered_after_retry:{total_attempts}" if total_attempts > 0 else None
        self._status = ProviderStatus(provider=self._provider_label, mode="live", reason=reason)
        return results

    @property
    def status(self) -> ProviderStatus:
        return self._status


def _hash_embed(text: str, size: int = 128) -> list[float]:
    values = [0.0] * size
    tokens = text.lower().split()
    if not tokens:
        return values

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index, byte in enumerate(digest[:16]):
            slot = (byte + index * 17) % size
            values[slot] += 1.0 if byte % 2 == 0 else -1.0

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _fallback_reason(exc: Exception, attempts: int, max_retries: int) -> str:
    class_name = exc.__class__.__name__
    if is_transient_provider_error(exc):
        return (
            f"provider_transient_error:{class_name}:retry_exhausted:{attempts}:"
            f"max_retries:{max_retries}"
        )
    return provider_error_reason(exc)
