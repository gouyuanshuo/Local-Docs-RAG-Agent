from __future__ import annotations

from abc import ABC, abstractmethod


class ChatProvider(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
