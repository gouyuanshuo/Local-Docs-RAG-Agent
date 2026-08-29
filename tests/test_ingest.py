from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import DocumentChunk, ProviderStatus
from local_docs_rag_agent.rag import ingest


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(provider="fake", mode="live")


class FakeQdrantStore:
    def __init__(self, collection_exists: bool) -> None:
        self._collection_exists = collection_exists
        self.saved_chunks: list[DocumentChunk] = []
        self.removed_source_paths: list[str] = []
        self.replaced_source_paths: list[str] = []

    def collection_exists(self) -> bool:
        return self._collection_exists

    def save(
        self,
        chunks: list[DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        self.saved_chunks = chunks
        self.removed_source_paths = removed_source_paths or []
        self.replaced_source_paths = replaced_source_paths or []


def _qdrant_config(tmp_path: Path, docs_dir: Path, manifest_path: Path) -> AppConfig:
    return AppConfig.from_env().with_overrides(
        docs_dir=docs_dir,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=manifest_path,
        vector_backend="qdrant",
        qdrant_url="https://qdrant.example",
        qdrant_collection="test-collection",
        chunk_strategy="markdown",
        chunk_size=800,
        chunk_overlap=120,
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    store: FakeQdrantStore,
) -> None:
    monkeypatch.setattr(ingest, "QdrantChunkStore", FakeQdrantStore)
    monkeypatch.setattr(ingest, "build_embedding_provider", lambda config: FakeEmbeddingProvider())
    monkeypatch.setattr(
        ingest,
        "build_store",
        lambda config, embedding_provider=None: store,
    )


def test_missing_collection_reingests_unchanged_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_text = "# Attention\n\nAttention uses queries, keys, and values."
    source_path.write_text(source_text, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    source_path.as_posix(): {
                        "checksum": ingest._checksum(source_text),
                        "chunk_ids": ["old-chunk"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=False)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(_qdrant_config(tmp_path, docs_dir, manifest_path))

    assert chunks
    assert store.saved_chunks == chunks
    assert store.replaced_source_paths == [source_path.as_posix()]


def test_existing_collection_skips_unchanged_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_text = "unchanged document"
    source_path.write_text(source_text, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    source_path.as_posix(): {
                        "checksum": ingest._checksum(source_text),
                        "chunk_ids": ["existing-chunk"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(_qdrant_config(tmp_path, docs_dir, manifest_path))

    assert chunks == []
    assert store.saved_chunks == []
    assert store.replaced_source_paths == []


def test_incremental_ingest_deletes_stale_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    removed_source = (docs_dir / "removed.md").as_posix()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    removed_source: {
                        "checksum": "old-checksum",
                        "chunk_ids": ["old-chunk"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(_qdrant_config(tmp_path, docs_dir, manifest_path))

    assert chunks == []
    assert store.removed_source_paths == [removed_source]
