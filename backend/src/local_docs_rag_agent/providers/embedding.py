from __future__ import annotations

import hashlib
import math
import time

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.providers.base import EmbeddingProvider


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
        self._client = self._build_client(api_key=api_key, base_url=base_url) if api_key else None
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_ms = max(0, retry_backoff_ms)
        if not api_key:
            self._status = ProviderStatus(provider=self._provider_label, mode="fallback", reason="missing_api_key")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._client:
            return [_hash_embed(text) for text in texts]

        results: list[list[float]] = []
        total_attempts = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            request: dict[str, object] = {
                "model": self._model,
                "input": batch,
            }
            if self._dimensions is not None:
                request["dimensions"] = self._dimensions

            attempt = 0
            while True:
                try:
                    response = self._client.embeddings.create(**request)
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
                    if _is_transient_embedding_error(exc) and attempt < self._max_retries:
                        sleep_ms = self._retry_backoff_ms * (2**attempt)
                        if sleep_ms > 0:
                            time.sleep(sleep_ms / 1000.0)
                        attempt += 1
                        continue
                    self._status = ProviderStatus(
                        provider=self._provider_label,
                        mode="fallback",
                        reason=_fallback_reason(exc=exc, attempts=attempt + 1, max_retries=self._max_retries),
                    )
                    return [_hash_embed(text) for text in texts]

        reason = f"recovered_after_retry:{total_attempts}" if total_attempts > 0 else None
        self._status = ProviderStatus(provider=self._provider_label, mode="live", reason=reason)
        return results

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def _build_client(self, api_key: str, base_url: str | None):
        try:
            from openai import DefaultHttpxClient, OpenAI
        except Exception:
            self._status = ProviderStatus(
                provider=self._provider_label,
                mode="fallback",
                reason="openai_client_unavailable",
            )
            return None

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if not self._trust_env:
            client_kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
        return OpenAI(**client_kwargs)


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


def _is_transient_embedding_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__
    if class_name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "temporarily",
            "try again",
            "connection reset",
            "connection aborted",
        )
    )


def _fallback_reason(exc: Exception, attempts: int, max_retries: int) -> str:
    class_name = exc.__class__.__name__
    if _is_transient_embedding_error(exc):
        return f"provider_transient_error:{class_name}:retry_exhausted:{attempts}:max_retries:{max_retries}"
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 240:
        message = message[:240].rstrip() + "..."
    return f"provider_error:{class_name}:{message}" if message else f"provider_error:{class_name}"
