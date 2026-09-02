"""Storage interface that every chunk backend implements.

Kept separate from the implementations so that `ingest`, the runtimes, and the tools
depend on the contract rather than on a concrete store, which is what allows the local
JSONL index and Qdrant to be swapped by configuration alone.
"""

from __future__ import annotations

from typing import Protocol

from local_docs_rag_agent.models import DocumentChunk, ProviderStatus, RetrievalHit


class ChunkStore(Protocol):
    """Persists document chunks and answers similarity queries over them."""

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        """Persist `chunks`, dropping points for removed and replaced sources.

        A backend that rewrites its whole index on every save may ignore both path
        lists; an incremental backend must honour them so stale points cannot survive.
        """
        ...

    def load(self) -> list[DocumentChunk]:
        """Return every stored chunk."""
        ...

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Return at most `top_k` hits for `query`, best first."""
        ...

    @property
    def embedding_status(self) -> ProviderStatus:
        """Report the health of the embedding provider backing this store."""
        ...
