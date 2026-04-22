from __future__ import annotations

import hashlib
import math

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.providers.base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str,
        dimensions: int | None,
        provider_label: str = "embedding",
    ) -> None:
        self._provider_label = provider_label.lower()
        self._status = ProviderStatus(provider=self._provider_label, mode="ready")
        self._client = self._build_client(api_key=api_key, base_url=base_url) if api_key else None
        self._model = model
        self._dimensions = dimensions
        if not api_key:
            self._status = ProviderStatus(provider=self._provider_label, mode="fallback", reason="missing_api_key")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._client:
            return [_hash_embed(text) for text in texts]

        request: dict[str, object] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions

        try:
            response = self._client.embeddings.create(**request)
            self._status = ProviderStatus(provider=self._provider_label, mode="live")
            return [item.embedding for item in response.data]
        except Exception as exc:
            self._status = ProviderStatus(
                provider=self._provider_label,
                mode="fallback",
                reason=f"provider_error:{exc.__class__.__name__}",
            )
            return [_hash_embed(text) for text in texts]

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def _build_client(self, api_key: str, base_url: str | None):
        try:
            from openai import OpenAI
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
