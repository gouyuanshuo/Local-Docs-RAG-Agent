"""The second retrieval stage: reordering a candidate window before it is answered.

First-stage retrieval optimises for recall. BM25 and cosine similarity both
judge a chunk without ever reading the question as a question — they match terms
and directions, not intent — so the right passage regularly lands third or
seventh rather than first. Widening `top_k` to compensate is the wrong fix: it
pushes more marginal context into the prompt, which is exactly what degrades an
answer.

A reranker is the other half of that trade. Retrieval proposes a wide candidate
window, the reranker reads the question and the candidates together and returns
a short, reordered list, and only that list reaches the model.

Two rules keep the stage honest:

* It is a `Protocol`, not a base class with an LLM in it. A cross-encoder
  reranker is the obvious next implementation, and it should need nothing from
  this module but the three members below.
* It never raises and never invents results. A reranker that cannot reach its
  model returns the candidates it was given, unchanged, and reports `fallback` —
  the same contract the chat and embedding providers follow, so a degraded run
  stays visibly degraded instead of quietly passing for a reranked one.

`candidate_depth` exists so that turning reranking off costs nothing. The
disabled reranker asks for exactly `top_k` candidates, which is the query the
project made before this stage existed; only a real reranker widens the window
it will pay to read.
"""

from __future__ import annotations

from typing import Protocol

from local_docs_rag_agent import models


class Reranker(Protocol):
    """Reorders first-stage candidates and reports whether it actually ran."""

    def candidate_depth(self, top_k: int) -> int:
        """Return how many first-stage candidates to retrieve for a `top_k` answer."""
        ...

    def rerank(
        self, *, query: str, hits: list[models.RetrievalHit], top_k: int
    ) -> list[models.RetrievalHit]:
        """Return at most `top_k` of `hits`, best first. Must not raise."""
        ...

    @property
    def status(self) -> models.ProviderStatus:
        """Report whether this reranker ran, was skipped, or degraded."""
        ...


class IdentityReranker:
    """The `none` reranker: keeps the first-stage order and asks for nothing extra.

    This is the default, and it is a real object rather than a `None` check at
    the call site so that the retrieval pipeline has one shape regardless of
    configuration, and so diagnostics always carry a reranker status to report.
    """

    def candidate_depth(self, top_k: int) -> int:
        return top_k

    def rerank(
        self, *, query: str, hits: list[models.RetrievalHit], top_k: int
    ) -> list[models.RetrievalHit]:
        del query
        return hits[:top_k]

    @property
    def status(self) -> models.ProviderStatus:
        # `ready` rather than `live`: nothing was reordered, and nothing
        # degraded either.
        return models.ProviderStatus(
            provider="none", mode="ready", reason="reranker_disabled"
        )
