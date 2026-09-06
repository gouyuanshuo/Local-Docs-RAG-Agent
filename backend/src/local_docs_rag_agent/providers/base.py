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
    """Answers a question from supplied context, and reports its health."""

    @abc.abstractmethod
    def answer(self, question: str, context: str) -> str:
        """Answer `question` using only `context`.

        Args:
          question: The user's question.
          context: Retrieved evidence the answer must be drawn from.

        Returns:
          The answer text. An implementation that cannot reach a live model
          returns a degraded answer and says so through `status` rather
          than raising.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> models.ProviderStatus:
        """Report whether this provider is live, degraded, or unused."""
        raise NotImplementedError


class EmbeddingProvider(abc.ABC):
    """Turns texts into vectors, and reports its health."""

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed every text in `texts`.

        Args:
          texts: The texts to embed.

        Returns:
          One vector per input, in input order. An implementation that
          falls back to synthetic vectors must report `fallback` through
          `status`, because ingest refuses to store them.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self) -> models.ProviderStatus:
        """Report whether this provider is live, degraded, or unused."""
        raise NotImplementedError
