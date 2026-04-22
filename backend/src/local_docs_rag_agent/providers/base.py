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
