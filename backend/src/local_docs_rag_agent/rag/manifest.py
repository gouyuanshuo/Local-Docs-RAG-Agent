"""Typed record of what the previous ingest wrote.

The manifest stores a checksum and the chunk ids for every source, plus the
fingerprint of the retrieval settings the index was built under. Together these
are what make an incremental ingest possible: unchanged sources are skipped,
removed sources have their points deleted, and a settings change invalidates the
whole index at once.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from local_docs_rag_agent import exceptions, models
from local_docs_rag_agent.rag import file_io


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One source document's checksum and the chunk ids it produced."""

    checksum: str
    chunk_ids: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, source_path: str, payload: object) -> ManifestEntry:
        """Rebuild an entry from its stored mapping.

        Args:
          source_path: The document this entry describes, used only to
            name the file in an error message.
          payload: The stored mapping.

        Returns:
          The reconstructed entry.

        Raises:
          DataFormatError: If the payload is not an object, or its
            checksum or chunk ids have the wrong type.
        """
        if not isinstance(payload, dict):
            raise exceptions.DataFormatError(
                f"Manifest entry for {source_path!r} must be an object"
            )
        checksum = payload.get("checksum")
        chunk_ids = payload.get("chunk_ids", [])
        if not isinstance(checksum, str):
            raise exceptions.DataFormatError(
                f"Manifest entry for {source_path!r} has an invalid checksum"
            )
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in chunk_ids
        ):
            raise exceptions.DataFormatError(
                f"Manifest entry for {source_path!r} has invalid chunk_ids"
            )
        return cls(checksum=checksum, chunk_ids=tuple(chunk_ids))

    def to_payload(self) -> dict[str, object]:
        """Return the entry as a JSON-serializable mapping."""
        return {"checksum": self.checksum, "chunk_ids": list(self.chunk_ids)}


@dataclasses.dataclass(frozen=True, slots=True)
class IngestManifest:
    """What the last ingest wrote, and under which settings.

    The fingerprint is what makes an incremental ingest safe: it
    changes whenever a setting changes what a stored vector means, so a
    reconfigured index is rebuilt rather than silently mixed.
    """

    sources: dict[str, ManifestEntry] = dataclasses.field(default_factory=dict)
    index_fingerprint: str | None = None

    @classmethod
    def load(cls, path: pathlib.Path) -> IngestManifest:
        """Read a manifest, treating a missing file as an empty one.

        Args:
          path: Where the manifest is stored.

        Returns:
          The stored manifest, or an empty one when no file exists yet.

        Raises:
          DataFormatError: If the file exists but cannot be read or does
            not hold a valid manifest.
        """
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise exceptions.DataFormatError(
                f"Could not read ingest manifest {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("sources"), dict
        ):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} must contain a sources object"
            )
        index_fingerprint = payload.get("index_fingerprint")
        if index_fingerprint is not None and not isinstance(
            index_fingerprint, str
        ):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} has an invalid index_fingerprint"
            )
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
        chunks: list[models.DocumentChunk],
        index_fingerprint: str,
    ) -> IngestManifest:
        """Return the manifest that describes the ingest just performed.

        Args:
          source_checksums: Current checksum of every discovered document.
          indexed_sources: The documents this run actually re-chunked.
          chunks: Every chunk this run produced.
          index_fingerprint: Fingerprint of the settings it ran under.

        Returns:
          A new manifest. A document that was not re-chunked keeps the
          chunk ids it already had, so an incremental run does not forget
          what it left in place.
        """
        chunk_ids_by_source: dict[str, list[str]] = {}
        for chunk in chunks:
            chunk_ids_by_source.setdefault(chunk.source_path, []).append(
                chunk.chunk_id
            )

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

    def save(self, path: pathlib.Path) -> None:
        """Write the manifest to `path`, atomically and with sorted keys."""
        payload = {
            "index_fingerprint": self.index_fingerprint,
            "sources": {
                source_path: entry.to_payload()
                for source_path, entry in sorted(self.sources.items())
            },
        }
        file_io.atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=True, indent=2),
        )
