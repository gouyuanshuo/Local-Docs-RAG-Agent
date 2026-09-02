import json

import pytest

from local_docs_rag_agent.exceptions import DataFormatError
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus
from local_docs_rag_agent.rag.local_store import LocalJsonlChunkStore


class StubEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(provider="stub", mode="fallback")


def test_local_store_round_trip_uses_no_leftover_temporary_file(tmp_path) -> None:
    index_path = tmp_path / "index" / "chunks.jsonl"
    chunk = DocumentChunk(
        chunk_id="one",
        source_path="doc.md",
        title="Doc",
        text="content",
        chunk_index=0,
        start_char=0,
        end_char=7,
        embedding=[1.0],
    )
    store = LocalJsonlChunkStore(index_path, StubEmbeddingProvider())

    store.save([chunk])

    assert store.load() == [chunk]
    assert list(index_path.parent.glob("*.tmp")) == []


def test_local_store_reports_corrupt_line_number(tmp_path) -> None:
    index_path = tmp_path / "chunks.jsonl"
    index_path.write_text(
        f"{json.dumps({'not': 'a chunk'})}\nnot-json\n",
        encoding="utf-8",
    )
    store = LocalJsonlChunkStore(index_path, StubEmbeddingProvider())

    with pytest.raises(DataFormatError, match="line 1"):
        store.load()
