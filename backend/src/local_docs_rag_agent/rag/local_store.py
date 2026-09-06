"""Chunk store backed by a single JSONL file on disk.

The file is human-readable and rewritten atomically on every ingest, which makes
local development inspectable and keeps a crashed run from leaving a
half-written index. Ranking itself lives in `rag.retrieval`, so which signals
are combined and how is a configured strategy rather than a property of the
storage format.
"""

from __future__ import annotations

import json
import pathlib

from local_docs_rag_agent import exceptions, models
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.rag import file_io, retrieval


class LocalJsonlChunkStore:
    """Small, inspectable on-disk store for local development and fallback use."""

    def __init__(
        self,
        index_path: pathlib.Path,
        embedding_provider: provider_base.EmbeddingProvider,
        settings: retrieval.RetrievalSettings | None = None,
    ) -> None:
        self._index_path = index_path
        self._embedding_provider = embedding_provider
        self._settings = settings or retrieval.RetrievalSettings()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        # A full rewrite makes the removal lists redundant for this backend.
        del removed_source_paths, replaced_source_paths
        content = "".join(
            f"{json.dumps(chunk.to_dict(), ensure_ascii=True)}\n"
            for chunk in chunks
        )
        file_io.atomic_write_text(self._index_path, content)

    def load(self) -> list[models.DocumentChunk]:
        if not self._index_path.exists():
            return []
        chunks: list[models.DocumentChunk] = []
        line_number: int | str = "unknown"
        try:
            with self._index_path.open("r", encoding="utf-8") as handle:
                for current_line_number, line in enumerate(handle, start=1):
                    line_number = current_line_number
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("chunk must be a JSON object")
                    chunks.append(models.DocumentChunk.from_dict(payload))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            exceptions.DataFormatError,
            TypeError,
            ValueError,
        ) as exc:
            raise exceptions.DataFormatError(
                f"Could not read local chunk index {self._index_path} "
                f"at line {line_number}: {exc}"
            ) from exc
        return chunks

    def search(self, query: str, top_k: int) -> list[models.RetrievalHit]:
        if top_k <= 0:
            return []
        query_vectors = self._embedding_provider.embed_texts([query])
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise exceptions.ProviderUnavailableError(
                "Embedding provider did not return one non-empty query vector"
            )
        # The whole corpus is available locally, so BM25 gets corpus-wide
        # document frequencies here rather than the candidate-window
        # approximation a remote backend has to settle for.
        return retrieval.rank_chunks(
            query=query,
            query_embedding=query_vectors[0],
            chunks=self.load(),
            top_k=top_k,
            settings=self._settings,
        )

    @property
    def embedding_status(self) -> models.ProviderStatus:
        return self._embedding_provider.status
