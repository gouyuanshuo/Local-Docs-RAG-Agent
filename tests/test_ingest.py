from __future__ import annotations

import json
import pathlib
import traceback
import types
from typing import Any

import pytest
import qdrant_client

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import rag
from local_docs_rag_agent.core import exceptions, models
from local_docs_rag_agent.providers import factory as provider_factory
from local_docs_rag_agent.rag import (
    index_lock,
    ingest,
    manifest,
    qdrant_store,
    store_factory,
)


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    @property
    def status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="live")


class FakeFallbackEmbeddingProvider(FakeEmbeddingProvider):
    @property
    def status(self) -> models.ProviderStatus:
        return models.ProviderStatus(provider="fake", mode="fallback")


class FakeIncrementalStore:
    """Non-Qdrant incremental store used to exercise the generic contract."""

    supports_incremental_updates = True
    requires_live_embeddings = True

    def __init__(self, *, exists: bool) -> None:
        self._exists = exists
        self.chunks_by_source: dict[str, list[models.DocumentChunk]] = {}
        self.save_history: list[
            tuple[list[models.DocumentChunk], list[str], list[str]]
        ] = []

    def exists(self) -> bool:
        return self._exists

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        removed = removed_source_paths or []
        replaced = replaced_source_paths or []
        self.save_history.append((list(chunks), list(removed), list(replaced)))
        for source_path in {*removed, *replaced}:
            self.chunks_by_source.pop(source_path, None)
        for chunk in chunks:
            self.chunks_by_source.setdefault(chunk.source_path, []).append(
                chunk
            )
        self._exists = True


class FakeQdrantStore:
    supports_incremental_updates = True
    requires_live_embeddings = True

    def __init__(self, collection_exists: bool) -> None:
        self._collection_exists = collection_exists
        self.save_calls = 0
        self.saved_chunks: list[models.DocumentChunk] = []
        self.removed_source_paths: list[str] = []
        self.replaced_source_paths: list[str] = []

    def collection_exists(self) -> bool:
        return self._collection_exists

    def exists(self) -> bool:
        return self.collection_exists()

    def save(
        self,
        chunks: list[models.DocumentChunk],
        removed_source_paths: list[str] | None = None,
        replaced_source_paths: list[str] | None = None,
    ) -> None:
        self.save_calls += 1
        self.saved_chunks = chunks
        self.removed_source_paths = removed_source_paths or []
        self.replaced_source_paths = replaced_source_paths or []


def _qdrant_config(
    tmp_path: pathlib.Path, docs_dir: pathlib.Path, manifest_path: pathlib.Path
) -> app_config.AppConfig:
    return app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs_dir,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=manifest_path,
        # Pinned so the index fingerprint records embedding_mode="live"
        # regardless of whether the machine running the test has embedding
        # credentials configured.
        embedding_api_key="test-embedding-key",
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
    monkeypatch.setattr(qdrant_store, "QdrantChunkStore", FakeQdrantStore)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: FakeEmbeddingProvider(),
    )
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )


def _install_incremental_fake(
    monkeypatch: pytest.MonkeyPatch,
    store: FakeIncrementalStore,
    provider: FakeEmbeddingProvider | None = None,
) -> None:
    selected_provider = provider or FakeEmbeddingProvider()
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: selected_provider,
    )
    monkeypatch.setattr(
        store_factory,
        "build_store",
        lambda config, embedding_provider=None: store,
    )


def _current_manifest_payload(
    config: app_config.AppConfig,
    sources: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "version": 2,
        "storage_identity": ingest.storage_identity(config),
        "index_fingerprint": ingest.index_fingerprint(
            config,
            embedding_mode="live",
        ),
        "sources": sources,
    }


def _local_incremental_config(
    tmp_path: pathlib.Path,
    docs_dir: pathlib.Path,
) -> app_config.AppConfig:
    return app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs_dir,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
        embedding_api_key="test-embedding-key",
        vector_backend="local",
        chunk_strategy="markdown",
        chunk_size=800,
        chunk_overlap=120,
    )


def test_third_incremental_store_obeys_all_source_transitions(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    first = docs_dir / "first.md"
    second = docs_dir / "second.md"
    first.write_text("First version", encoding="utf-8")
    second.write_text("Second version", encoding="utf-8")
    config = _local_incremental_config(tmp_path, docs_dir)
    config.ingest_manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    first.as_posix(): {
                        "checksum": ingest.source_checksum("First version"),
                        "chunk_ids": ["missing-first"],
                    },
                    second.as_posix(): {
                        "checksum": ingest.source_checksum("Second version"),
                        "chunk_ids": ["missing-second"],
                    },
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeIncrementalStore(exists=False)
    _install_incremental_fake(monkeypatch, store)

    missing = ingest.ingest_documents(config)
    assert {chunk.source_path for chunk in missing} == {
        first.as_posix(),
        second.as_posix(),
    }
    assert store.save_history[-1][1] == []
    assert store.save_history[-1][2] == [first.as_posix(), second.as_posix()]

    unchanged = ingest.ingest_documents(config)
    assert unchanged == []
    assert store.save_history[-1] == ([], [], [])

    first.write_text("First changed", encoding="utf-8")
    changed = ingest.ingest_documents(config)
    assert [chunk.source_path for chunk in changed] == [first.as_posix()]
    assert store.save_history[-1][2] == [first.as_posix()]

    second.unlink()
    removed = ingest.ingest_documents(config)
    assert removed == []
    assert store.save_history[-1][1] == [second.as_posix()]
    assert second.as_posix() not in store.chunks_by_source

    first.write_text("   \n", encoding="utf-8")
    emptied = ingest.ingest_documents(config)
    assert emptied == []
    assert store.save_history[-1][2] == [first.as_posix()]
    assert store.chunks_by_source == {}


def test_store_capability_requires_live_embeddings_independent_of_backend(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text("Evidence", encoding="utf-8")
    config = _local_incremental_config(tmp_path, docs_dir).with_overrides(
        embedding_api_key=None
    )
    store = FakeIncrementalStore(exists=False)
    _install_incremental_fake(
        monkeypatch,
        store,
        provider=FakeFallbackEmbeddingProvider(),
    )

    with pytest.raises(exceptions.ProviderUnavailableError):
        ingest.ingest_documents(config)

    assert store.save_history == []


def test_ensure_index_uses_store_existence_capability(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "sample.md"
    source.write_text("Evidence", encoding="utf-8")
    config = _local_incremental_config(tmp_path, docs_dir)
    config.index_path.write_text("unrelated sentinel", encoding="utf-8")
    config.ingest_manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    source.as_posix(): {
                        "checksum": ingest.source_checksum("Evidence"),
                        "chunk_ids": ["missing"],
                    }
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeIncrementalStore(exists=False)
    _install_incremental_fake(monkeypatch, store)

    ingest.ensure_index(config)

    assert store.exists() is True
    assert [chunk.source_path for chunk in store.save_history[-1][0]] == [
        source.as_posix()
    ]


def test_missing_collection_reingests_unchanged_manifest(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_text = "# Attention\n\nAttention uses queries, keys, and values."
    source_path.write_text(source_text, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)
    manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    source_path.as_posix(): {
                        "checksum": ingest.source_checksum(source_text),
                        "chunk_ids": ["old-chunk"],
                    }
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=False)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(config)

    assert chunks
    assert store.saved_chunks == chunks
    assert store.replaced_source_paths == [source_path.as_posix()]


def test_existing_collection_skips_unchanged_document(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_text = "unchanged document"
    source_path.write_text(source_text, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)
    manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    source_path.as_posix(): {
                        "checksum": ingest.source_checksum(source_text),
                        "chunk_ids": ["existing-chunk"],
                    }
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(config)

    assert chunks == []
    assert store.saved_chunks == []
    assert store.replaced_source_paths == []


def test_incremental_ingest_deletes_stale_source(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    removed_source = (docs_dir / "removed.md").as_posix()
    manifest_path = tmp_path / "manifest.json"
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)
    manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    removed_source: {
                        "checksum": "old-checksum",
                        "chunk_ids": ["old-chunk"],
                    }
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(config)

    assert chunks == []
    assert store.removed_source_paths == [removed_source]


def test_changed_document_that_becomes_empty_deletes_old_chunks(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_path.write_text("   \n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)
    manifest_path.write_text(
        json.dumps(
            _current_manifest_payload(
                config,
                {
                    source_path.as_posix(): {
                        "checksum": "old-checksum",
                        "chunk_ids": ["old-chunk"],
                    }
                },
            )
        ),
        encoding="utf-8",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    chunks = ingest.ingest_documents(config)

    assert chunks == []
    assert store.replaced_source_paths == [source_path.as_posix()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources"][source_path.as_posix()]["chunk_ids"] == []


def test_missing_docs_directory_does_not_overwrite_index_or_manifest(
    tmp_path: pathlib.Path,
) -> None:
    missing_docs = tmp_path / "missing"
    index_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    index_path.write_text("existing-index", encoding="utf-8")
    manifest_path.write_text("existing-manifest", encoding="utf-8")
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=missing_docs,
        index_path=index_path,
        ingest_manifest_path=manifest_path,
        vector_backend="local",
    )

    with pytest.raises(exceptions.ConfigurationError, match="does not exist"):
        ingest.ingest_documents(config)

    assert index_path.read_text(encoding="utf-8") == "existing-index"
    assert manifest_path.read_text(encoding="utf-8") == "existing-manifest"


def test_ensure_index_rebuilds_when_chunk_configuration_changes(
    tmp_path: pathlib.Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("a" * 80, encoding="utf-8")
    index_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs_dir,
        docs_exclude_patterns=[],
        index_path=index_path,
        ingest_manifest_path=manifest_path,
        vector_backend="local",
        chunk_strategy="fixed",
        chunk_size=100,
        chunk_overlap=10,
        embedding_api_key=None,
    )
    ingest.ingest_documents(config)
    assert len(index_path.read_text(encoding="utf-8").splitlines()) == 1

    ingest.ensure_index(config.with_overrides(chunk_size=30, chunk_overlap=5))

    assert len(index_path.read_text(encoding="utf-8").splitlines()) == 3


def test_excluding_all_documents_invalidates_index(
    tmp_path: pathlib.Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Private evidence", encoding="utf-8")
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
    )
    ingest.ingest_documents(config)
    excluded = config.with_overrides(docs_exclude_patterns=["*"])
    ingest.ensure_index(excluded)
    assert store_factory.build_store(excluded).load() == []


def test_changing_docs_directory_replaces_same_local_index_scope(
    tmp_path: pathlib.Path,
) -> None:
    first_docs = tmp_path / "first-docs"
    first_docs.mkdir()
    first_source = first_docs / "first.md"
    first_source.write_text("First corpus", encoding="utf-8")
    second_docs = tmp_path / "second-docs"
    second_docs.mkdir()
    second_source = second_docs / "second.md"
    second_source.write_text("Second corpus", encoding="utf-8")
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=first_docs,
        docs_exclude_patterns=[],
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
        embedding_api_key=None,
    )
    ingest.ingest_documents(config)

    changed = config.with_overrides(docs_dir=second_docs)
    ingest.ensure_index(changed)

    chunks = store_factory.build_store(changed).load()
    assert {chunk.source_path for chunk in chunks} == {second_source.as_posix()}
    stored = manifest.IngestManifest.load(changed.ingest_manifest_path)
    assert set(stored.sources) == {second_source.as_posix()}


def test_index_fingerprint_normalizes_exclusion_pattern_order(
    tmp_path: pathlib.Path,
) -> None:
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=tmp_path / "docs",
        docs_exclude_patterns=["drafts/*", "*.tmp", "drafts/*"],
    )
    reordered = config.with_overrides(
        docs_exclude_patterns=["*.tmp", "drafts/*"]
    )

    assert ingest.index_fingerprint(
        config, embedding_mode="fallback"
    ) == ingest.index_fingerprint(reordered, embedding_mode="fallback")


def test_switching_local_index_path_requires_dedicated_manifest(
    tmp_path: pathlib.Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Evidence", encoding="utf-8")
    original_index = tmp_path / "first.jsonl"
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=original_index,
        ingest_manifest_path=tmp_path / "manifest.json",
        embedding_api_key=None,
    )
    ingest.ingest_documents(config)
    original_content = original_index.read_text(encoding="utf-8")
    switched = config.with_overrides(index_path=tmp_path / "second.jsonl")

    with pytest.raises(
        exceptions.ConfigurationError,
        match="dedicated INGEST_MANIFEST_PATH",
    ):
        ingest.ensure_index(switched)

    assert original_index.read_text(encoding="utf-8") == original_content
    assert not switched.index_path.exists()


def test_local_index_dangling_symlink_survives_ingest_and_ensure(
    tmp_path: pathlib.Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Evidence", encoding="utf-8")
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    index_target = target_dir / "chunks.jsonl"
    index_link = tmp_path / "chunks-link.jsonl"
    index_link.symlink_to(index_target)
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=index_link,
        ingest_manifest_path=tmp_path / "manifest.json",
        embedding_api_key=None,
    )
    original_identity = ingest.storage_identity(config)

    ingest.ingest_documents(config)
    ingest.ensure_index(config)

    assert index_link.is_symlink()
    assert index_target.exists()
    assert ingest.storage_identity(config) == original_identity
    assert [
        chunk.text for chunk in store_factory.build_store(config).load()
    ] == ["Evidence"]


@pytest.mark.parametrize("target_exists", [False, True])
def test_manifest_final_symlink_survives_ingest_and_ensure(
    tmp_path: pathlib.Path,
    target_exists: bool,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Evidence", encoding="utf-8")
    index_dir = tmp_path / "index"
    manifest_link_dir = tmp_path / "manifest-link"
    manifest_target_dir = tmp_path / "manifest-target"
    index_dir.mkdir()
    manifest_link_dir.mkdir()
    manifest_target_dir.mkdir()
    manifest_target = manifest_target_dir / "manifest.json"
    if target_exists:
        manifest.IngestManifest().save(manifest_target)
    manifest_link = manifest_link_dir / "manifest.json"
    manifest_link.symlink_to(manifest_target)
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=index_dir / "chunks.jsonl",
        ingest_manifest_path=manifest_link,
        embedding_api_key=None,
    )

    ingest.ingest_documents(config)
    ingest.ensure_index(config)

    assert manifest_link.is_symlink()
    assert manifest_target.exists()
    assert (
        manifest.IngestManifest.load(manifest_target).repair_required is False
    )
    assert list(manifest_link_dir.glob("*.lock")) == []
    assert list(manifest_target_dir.glob("*.lock"))


@pytest.mark.parametrize("alias_kind", ["direct", "symlinks", "hardlinks"])
def test_local_index_and_manifest_may_not_alias_same_file(
    tmp_path: pathlib.Path,
    alias_kind: str,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Evidence", encoding="utf-8")
    shared_target = tmp_path / "shared-state.json"
    shared_target.write_bytes(b"unchanged-state")
    if alias_kind == "symlinks":
        index_path = tmp_path / "index-link.jsonl"
        manifest_path = tmp_path / "manifest-link.json"
        index_path.symlink_to(shared_target)
        manifest_path.symlink_to(shared_target)
    elif alias_kind == "hardlinks":
        index_path = tmp_path / "index-hardlink.jsonl"
        manifest_path = tmp_path / "manifest-hardlink.json"
        index_path.hardlink_to(shared_target)
        manifest_path.hardlink_to(shared_target)
    else:
        index_path = shared_target
        manifest_path = shared_target
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=index_path,
        ingest_manifest_path=manifest_path,
        embedding_api_key=None,
    )

    with pytest.raises(
        exceptions.ConfigurationError,
        match=r"INDEX_PATH.*INGEST_MANIFEST_PATH",
    ):
        ingest.ingest_documents(config)

    assert shared_target.read_bytes() == b"unchanged-state"
    assert index_path.exists()
    assert manifest_path.exists()
    if alias_kind == "symlinks":
        assert index_path.is_symlink()
        assert manifest_path.is_symlink()


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("qdrant_url", "https://other.example"),
        ("qdrant_collection", "other-collection"),
    ],
)
def test_switching_qdrant_target_never_replays_manifest_deletions(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
    value: str,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source.write_text("Evidence", encoding="utf-8")
    config = _qdrant_config(tmp_path, docs, tmp_path / "manifest.json")
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)
    ingest.ingest_documents(config)
    source.unlink()
    save_calls = store.save_calls

    switched = config.with_overrides(**{override: value})
    with pytest.raises(
        exceptions.ConfigurationError,
        match="dedicated INGEST_MANIFEST_PATH",
    ):
        ingest.ingest_documents(switched)

    assert store.save_calls == save_calls
    assert store.removed_source_paths == []


def test_same_qdrant_target_scope_change_deletes_newly_excluded_source(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source.write_text("Private evidence", encoding="utf-8")
    config = _qdrant_config(tmp_path, docs, tmp_path / "manifest.json")
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)
    ingest.ingest_documents(config)

    ingest.ensure_index(config.with_overrides(docs_exclude_patterns=["*"]))

    assert store.removed_source_paths == [source.as_posix()]
    assert store.save_calls == 2


def test_qdrant_storage_identity_canonicalizes_endpoint_credentials(
    tmp_path: pathlib.Path,
) -> None:
    docs = tmp_path / "docs"
    manifest_path = tmp_path / "manifest.json"
    first = _qdrant_config(tmp_path, docs, manifest_path).with_overrides(
        qdrant_url=(
            "https://url-user:url-password@QDRANT.EXAMPLE:6333/cluster/"
            "?api_key=one#first"
        ),
        qdrant_api_key="header-one",
    )
    second = first.with_overrides(
        qdrant_url="https://qdrant.example:6333/cluster?api_key=two#second",
        qdrant_api_key="header-two",
    )

    assert ingest.storage_identity(first) == ingest.storage_identity(second)


def test_qdrant_storage_identity_treats_omitted_port_as_6333(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url="https://qdrant.example/cluster")
    explicit = config.with_overrides(
        qdrant_url="https://qdrant.example:6333/cluster"
    )

    assert ingest.storage_identity(config) == ingest.storage_identity(explicit)


def test_qdrant_ipv6_storage_identity_treats_omitted_port_as_6333(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url="https://[2001:db8::1]/cluster")
    explicit = config.with_overrides(
        qdrant_url="https://[2001:db8::1]:6333/cluster"
    )

    assert ingest.storage_identity(config) == ingest.storage_identity(explicit)


def test_qdrant_storage_identity_normalizes_equivalent_ipv6_literals(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(
        qdrant_url=("https://[2001:0DB8:0000:0000:0000:0000:0000:0001]/cluster")
    )
    compressed = config.with_overrides(
        qdrant_url="https://[2001:db8::1]:6333/cluster"
    )

    assert ingest.storage_identity(config) == ingest.storage_identity(
        compressed
    )


@pytest.mark.parametrize(("scheme", "port"), [("http", 80), ("https", 443)])
def test_qdrant_standard_web_port_switch_requires_dedicated_manifest(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
    port: int,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source.write_text("Evidence", encoding="utf-8")
    config = _qdrant_config(
        tmp_path,
        docs,
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url=f"{scheme}://qdrant.example/cluster")
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)
    ingest.ingest_documents(config)
    source.unlink()
    save_calls = store.save_calls

    switched = config.with_overrides(
        qdrant_url=f"{scheme}://qdrant.example:{port}/cluster"
    )
    with pytest.raises(
        exceptions.ConfigurationError,
        match="dedicated INGEST_MANIFEST_PATH",
    ):
        ingest.ingest_documents(switched)

    assert store.save_calls == save_calls
    assert store.removed_source_paths == []


def test_qdrant_storage_identity_rejects_port_zero(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url="https://qdrant.example:0/cluster")

    with pytest.raises(
        exceptions.ConfigurationError,
        match=r"valid HTTP\(S\) endpoint",
    ):
        ingest.storage_identity(config)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://fake-user:fake-password@example.invalid\uff0fbad",
        "http://fake-user:fake-password@[2001:db8::1",
    ],
)
def test_malformed_qdrant_url_traceback_excludes_credentials(
    tmp_path: pathlib.Path,
    bad_url: str,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url=bad_url)

    with pytest.raises(
        exceptions.ConfigurationError,
        match=r"valid HTTP\(S\) endpoint",
    ) as exc_info:
        ingest.storage_identity(config)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "fake-user" not in formatted
    assert "fake-password" not in formatted
    assert exc_info.value.__suppress_context__ is True


def test_validate_storage_target_is_side_effect_free(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    config = _qdrant_config(
        tmp_path,
        docs,
        tmp_path / "manifest.json",
    )

    def unexpected_call(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("validation must not construct or lock")

    monkeypatch.setattr(qdrant_client, "QdrantClient", unexpected_call)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        unexpected_call,
    )
    monkeypatch.setattr(store_factory, "build_store", unexpected_call)
    monkeypatch.setattr(index_lock, "index_guard", unexpected_call)

    rag.validate_storage_target(config)

    assert not docs.exists()
    assert not config.index_path.exists()
    assert not config.ingest_manifest_path.exists()


def test_validate_storage_target_rejects_missing_qdrant_url(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(qdrant_url=None)

    with pytest.raises(exceptions.ConfigurationError, match="QDRANT_URL"):
        rag.validate_storage_target(config)


def test_validate_storage_target_rejects_unknown_backend(
    tmp_path: pathlib.Path,
) -> None:
    config = app_config.AppConfig.from_env().with_overrides(
        index_path=tmp_path / "index.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
    )
    object.__setattr__(config, "vector_backend", "future-store")

    with pytest.raises(exceptions.ConfigurationError, match="unsupported"):
        rag.validate_storage_target(config)


def test_validate_storage_target_malformed_url_traceback_excludes_credentials(
    tmp_path: pathlib.Path,
) -> None:
    config = _qdrant_config(
        tmp_path,
        tmp_path / "docs",
        tmp_path / "manifest.json",
    ).with_overrides(
        qdrant_url=(
            "http://validation-user:validation-password@"
            "example.invalid\uff0fbad"
        )
    )

    with pytest.raises(exceptions.ConfigurationError) as exc_info:
        rag.validate_storage_target(config)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "validation-user" not in formatted
    assert "validation-password" not in formatted


def test_manifest_storage_identity_does_not_persist_qdrant_credentials(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("Evidence", encoding="utf-8")
    config = _qdrant_config(
        tmp_path,
        docs,
        tmp_path / "manifest.json",
    ).with_overrides(
        qdrant_url=(
            "https://url-user:url-password@qdrant.example/cluster/"
            "?api_key=query-secret#fragment"
        ),
        qdrant_api_key="header-secret",
    )
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    ingest.ingest_documents(config)

    manifest_text = config.ingest_manifest_path.read_text(encoding="utf-8")
    stored = manifest.IngestManifest.load(config.ingest_manifest_path)
    assert stored.version == 2
    assert stored.storage_identity is not None
    assert len(stored.storage_identity) == 64
    assert config.qdrant_url is not None
    for secret in (
        "url-user",
        "url-password",
        "query-secret",
        "header-secret",
        config.qdrant_url,
    ):
        assert secret not in manifest_text


def test_legacy_local_manifest_rebuilds_and_acquires_storage_identity(
    tmp_path: pathlib.Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source_text = "Current evidence"
    source.write_text(source_text, encoding="utf-8")
    index_path = tmp_path / "chunks.jsonl"
    index_path.write_text("stale-index", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs,
        docs_exclude_patterns=[],
        index_path=index_path,
        ingest_manifest_path=manifest_path,
        embedding_api_key=None,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "index_fingerprint": ingest.index_fingerprint(
                    config,
                    embedding_mode="fallback",
                ),
                "sources": {
                    source.as_posix(): {
                        "checksum": ingest.source_checksum(source_text),
                        "chunk_ids": ["stale-chunk"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    ingest.ensure_index(config)

    stored_chunks = store_factory.build_store(config).load()
    assert [chunk.text for chunk in stored_chunks] == [source_text]
    stored = manifest.IngestManifest.load(manifest_path)
    assert stored.version == 2
    assert stored.storage_identity == ingest.storage_identity(config)


def test_legacy_qdrant_manifest_requires_ownership_decision(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "sample.md"
    source.write_text("Evidence", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": {
                    source.as_posix(): {
                        "checksum": "legacy-checksum",
                        "chunk_ids": ["legacy-chunk"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = _qdrant_config(tmp_path, docs, manifest_path)
    store = FakeQdrantStore(collection_exists=True)
    _install_fakes(monkeypatch, store)

    with pytest.raises(
        exceptions.ConfigurationError,
        match="legacy Qdrant manifest",
    ):
        ingest.ingest_documents(config)

    assert store.save_calls == 0


class _RecordingQdrantClient:
    """In-process Qdrant stand-in that can fail after delete."""

    def __init__(self) -> None:
        self.exists = False
        self.fail_upsert = False
        self.upserts = 0
        self.deletes = 0

    def collection_exists(self, name: str) -> bool:
        del name
        return self.exists

    def create_collection(self, **kwargs: Any) -> None:
        del kwargs
        self.exists = True

    def create_payload_index(self, **kwargs: Any) -> None:
        del kwargs

    def delete(self, **kwargs: Any) -> None:
        del kwargs
        self.deletes += 1

    def upsert(self, **kwargs: Any) -> None:
        del kwargs
        self.upserts += 1
        if self.fail_upsert:
            raise RuntimeError("forced upsert failure")
        self.exists = True

    def get_collection(self, name: str) -> types.SimpleNamespace:
        del name
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                params=types.SimpleNamespace(
                    vectors=types.SimpleNamespace(size=2)
                )
            )
        )


def test_qdrant_upsert_failure_reindexes_unchanged_checksum(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_path.write_text("Attention uses queries and keys.", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    client = _RecordingQdrantClient()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: FakeEmbeddingProvider(),
    )
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)

    first = ingest.ingest_documents(config)
    assert first
    assert isinstance(
        store_factory.build_store(config), qdrant_store.QdrantChunkStore
    )
    assert client.upserts == 1
    stored = manifest.IngestManifest.load(manifest_path)
    assert stored.needs_reindex == ()

    client.exists = False
    client.fail_upsert = True
    upserts_before_fail = client.upserts
    with pytest.raises(exceptions.VectorStoreError):
        ingest.ingest_documents(config)
    assert client.upserts == upserts_before_fail + 1
    dirty = manifest.IngestManifest.load(manifest_path)
    assert source_path.as_posix() in dirty.needs_reindex
    assert (
        dirty.sources[source_path.as_posix()].checksum
        == stored.sources[source_path.as_posix()].checksum
    )

    client.exists = True
    client.fail_upsert = False
    upserts_before_repair = client.upserts
    repaired = ingest.ingest_documents(config)
    assert repaired
    assert client.upserts == upserts_before_repair + 1
    clean = manifest.IngestManifest.load(manifest_path)
    assert clean.needs_reindex == ()


def test_ensure_index_restores_qdrant_after_upsert_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ask/eval call ensure_index, not ingest_documents. Fingerprint and
    # checksums still match after a failed upsert, so skipping
    # needs_reindex would leave the collection empty.
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "sample.md"
    source_path.write_text("Attention uses queries and keys.", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    client = _RecordingQdrantClient()
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        lambda config: FakeEmbeddingProvider(),
    )
    config = _qdrant_config(tmp_path, docs_dir, manifest_path)

    ingest.ingest_documents(config)
    client.exists = False
    client.fail_upsert = True
    with pytest.raises(exceptions.VectorStoreError):
        ingest.ingest_documents(config)
    dirty = manifest.IngestManifest.load(manifest_path)
    assert source_path.as_posix() in dirty.needs_reindex

    client.exists = True
    client.fail_upsert = False
    upserts_before_ensure = client.upserts
    ingest.ensure_index(config)
    assert client.upserts == upserts_before_ensure + 1
    restored = manifest.IngestManifest.load(manifest_path)
    assert restored.needs_reindex == ()
