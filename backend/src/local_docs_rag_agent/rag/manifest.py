from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from local_docs_rag_agent.exceptions import DataFormatError
from local_docs_rag_agent.models import DocumentChunk
from local_docs_rag_agent.rag.file_io import atomic_write_text


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    checksum: str
    chunk_ids: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, source_path: str, payload: object) -> ManifestEntry:
        if not isinstance(payload, dict):
            raise DataFormatError(f"Manifest entry for {source_path!r} must be an object")
        checksum = payload.get("checksum")
        chunk_ids = payload.get("chunk_ids", [])
        if not isinstance(checksum, str):
            raise DataFormatError(f"Manifest entry for {source_path!r} has an invalid checksum")
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in chunk_ids
        ):
            raise DataFormatError(f"Manifest entry for {source_path!r} has invalid chunk_ids")
        return cls(checksum=checksum, chunk_ids=tuple(chunk_ids))

    def to_payload(self) -> dict[str, object]:
        return {"checksum": self.checksum, "chunk_ids": list(self.chunk_ids)}


@dataclass(frozen=True, slots=True)
class IngestManifest:
    sources: dict[str, ManifestEntry] = field(default_factory=dict)
    index_fingerprint: str | None = None

    @classmethod
    def load(cls, path: Path) -> IngestManifest:
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataFormatError(f"Could not read ingest manifest {path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
            raise DataFormatError(f"Ingest manifest {path} must contain a sources object")
        index_fingerprint = payload.get("index_fingerprint")
        if index_fingerprint is not None and not isinstance(index_fingerprint, str):
            raise DataFormatError(f"Ingest manifest {path} has an invalid index_fingerprint")
        return cls(
            sources={
                str(source_path): ManifestEntry.from_payload(
                    str(source_path),
                    entry_payload,
                )
                for source_path, entry_payload in payload["sources"].items()
            },
            index_fingerprint=index_fingerprint,
        )

    def updated(
        self,
        *,
        source_checksums: dict[str, str],
        indexed_sources: set[str],
        chunks: list[DocumentChunk],
        index_fingerprint: str,
    ) -> IngestManifest:
        chunk_ids_by_source: dict[str, list[str]] = {}
        for chunk in chunks:
            chunk_ids_by_source.setdefault(chunk.source_path, []).append(chunk.chunk_id)

        next_sources: dict[str, ManifestEntry] = {}
        for source_path, checksum in source_checksums.items():
            if source_path in indexed_sources:
                chunk_ids = tuple(chunk_ids_by_source.get(source_path, []))
            else:
                existing = self.sources.get(source_path)
                chunk_ids = existing.chunk_ids if existing else ()
            next_sources[source_path] = ManifestEntry(
                checksum=checksum,
                chunk_ids=chunk_ids,
            )
        return IngestManifest(
            sources=next_sources,
            index_fingerprint=index_fingerprint,
        )

    def save(self, path: Path) -> None:
        payload = {
            "index_fingerprint": self.index_fingerprint,
            "sources": {
                source_path: entry.to_payload()
                for source_path, entry in sorted(self.sources.items())
            },
        }
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=True, indent=2),
        )
