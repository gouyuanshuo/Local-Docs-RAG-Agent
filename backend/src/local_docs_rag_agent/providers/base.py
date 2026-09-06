"""Chat and embedding interfaces every provider implements.

Both interfaces require a `status` property. A provider that degrades must say
so there rather than raising or silently returning a lesser result, which is
what allows an ingest to refuse fallback vectors and a live check to reject a
degraded run.
"""

from __future__ import annotations

import abc

from local_docs_rag_agent import models


class ChatProvider(abc.ABC):
    @abc.abstractmethod
    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> models.ProviderStatus:
        raise NotImplementedError


class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> models.ProviderStatus:
        raise NotImplementedError
