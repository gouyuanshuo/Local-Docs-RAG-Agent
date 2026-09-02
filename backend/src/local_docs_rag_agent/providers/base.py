"""Chat and embedding interfaces every provider implements.

Both interfaces require a `status` property. A provider that degrades must say so there
rather than raising or silently returning a lesser result, which is what allows an
ingest to refuse fallback vectors and a live check to reject a degraded run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from local_docs_rag_agent.models import ProviderStatus


class ChatProvider(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        raise NotImplementedError
