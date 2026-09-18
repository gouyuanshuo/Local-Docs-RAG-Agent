"""Versioned record of what the previous ingest wrote and where it wrote it.

The manifest stores a checksum and the chunk ids for every source, plus the
fingerprint of the scope and retrieval settings the index was built under. Its
hashed storage identity keeps one target's source ownership from being replayed
against another target. A durable repair marker is published before store
mutation and cleared only after success. Together these fields make an
incremental ingest safe: unchanged sources are skipped, removed sources have
their points deleted, and interrupted writes are repaired before retrieval.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from local_docs_rag_agent.core import exceptions, file_io, models

CURRENT_VERSION = 2
LEGACY_VERSION = 1


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

    The fingerprint changes whenever document scope or a setting changes what
    a stored vector means. The storage identity separately proves which local
    file or Qdrant collection owns the recorded deletion list. `repair_required`
    means a store mutation began without a subsequent clean publication.
    """

    version: int = CURRENT_VERSION
    storage_identity: str | None = None
    sources: dict[str, ManifestEntry] = dataclasses.field(default_factory=dict)
    index_fingerprint: str | None = None
    needs_reindex: tuple[str, ...] = ()
    repair_required: bool = False

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
        version = payload.get("version", LEGACY_VERSION)
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version not in {LEGACY_VERSION, CURRENT_VERSION}
        ):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} has an unsupported version"
            )
        storage_identity = payload.get("storage_identity")
        if storage_identity is not None and (
            not isinstance(storage_identity, str) or not storage_identity
        ):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} has an invalid storage_identity"
            )
        index_fingerprint = payload.get("index_fingerprint")
        if index_fingerprint is not None and not isinstance(
            index_fingerprint, str
        ):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} has an invalid index_fingerprint"
            )
        needs_reindex = _string_tuple(
            payload.get("needs_reindex", []), path, "needs_reindex"
        )
        repair_required = payload.get("repair_required", False)
        if not isinstance(repair_required, bool):
            raise exceptions.DataFormatError(
                f"Ingest manifest {path} has an invalid repair_required"
            )
        return cls(
            version=version,
            storage_identity=storage_identity,
            sources={
                str(source_path): ManifestEntry.from_payload(
                    str(source_path),
                    entry_payload,
                )
                for source_path, entry_payload in payload["sources"].items()
            },
            index_fingerprint=index_fingerprint,
            needs_reindex=needs_reindex,
            repair_required=repair_required,
        )

    def updated(
        self,
        *,
        source_checksums: dict[str, str],
        indexed_sources: set[str],
        chunks: list[models.DocumentChunk],
        index_fingerprint: str,
        storage_identity: str,
    ) -> IngestManifest:
        """Return the manifest that describes the ingest just performed.

        Args:
          source_checksums: Current checksum of every discovered document.
          indexed_sources: The documents this run actually re-chunked.
          chunks: Every chunk this run produced.
          index_fingerprint: Fingerprint of the settings it ran under.
          storage_identity: Hashed identity of the store this run wrote.

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
        remaining_dirty = tuple(
            sorted(
                source_path
                for source_path in self.needs_reindex
                if source_path in next_sources
                and source_path not in indexed_sources
            )
        )
        return IngestManifest(
            version=CURRENT_VERSION,
            storage_identity=storage_identity,
            sources=next_sources,
            index_fingerprint=index_fingerprint,
            needs_reindex=remaining_dirty,
            repair_required=False,
        )

    def marked_needs_reindex(
        self,
        source_paths: tuple[str, ...],
        *,
        storage_identity: str,
    ) -> IngestManifest:
        """Return a copy that will reindex `source_paths` on the next ingest.

        Published before a store mutation so an interrupted write cannot let
        matching checksums be treated as "already indexed".

        Args:
          source_paths: Sources whose Qdrant points may be missing.
          storage_identity: Hashed identity of the store about to be mutated.

        Returns:
          A new manifest. Existing checksums are left in place so the
          next ingest can still see which file it is repairing.
        """
        return IngestManifest(
            version=CURRENT_VERSION,
            storage_identity=storage_identity,
            sources=self.sources,
            index_fingerprint=self.index_fingerprint,
            needs_reindex=tuple(
                sorted(set(self.needs_reindex) | set(source_paths))
            ),
            repair_required=True,
        )

    def save(self, path: pathlib.Path) -> None:
        """Write the manifest to `path`, atomically and with sorted keys."""
        payload = {
            "version": CURRENT_VERSION,
            "storage_identity": self.storage_identity,
            "index_fingerprint": self.index_fingerprint,
            "needs_reindex": list(self.needs_reindex),
            "repair_required": self.repair_required,
            "sources": {
                source_path: entry.to_payload()
                for source_path, entry in sorted(self.sources.items())
            },
        }
        file_io.atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


def _string_tuple(
    value: object, path: pathlib.Path, field_name: str
) -> tuple[str, ...]:
    """Return `value` as a tuple of strings.

    Args:
      value: Stored JSON value.
      path: Manifest path, used in the error message.
      field_name: Field being parsed.

    Returns:
      The strings, dropping blanks.

    Raises:
      DataFormatError: If `value` is not a list of strings.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise exceptions.DataFormatError(
            f"Ingest manifest {path} has an invalid {field_name}"
        )
    return tuple(item for item in value if item)
