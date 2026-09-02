"""Finds and reads the source documents that feed the retrieval index.

Discovery is separated from ingestion because it is also the answer to "what can this
agent see?" — `tools.list_documents` and the `/api/documents` endpoint use it without
touching the index at all. Keeping it here means the file set an answer is grounded in
and the file set reported to the user can never disagree.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from local_docs_rag_agent.exceptions import ConfigurationError

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})


def collect_document_paths(
    docs_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Return every supported document under `docs_dir`, sorted and filtered.

    Sorting keeps chunk ids stable across runs on different filesystems, which is what
    lets the ingest manifest recognise an unchanged document.
    """

    require_docs_directory(docs_dir)
    patterns = exclude_patterns or []
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_excluded(path, docs_dir, patterns)
    )


def read_source_texts(
    docs_dir: Path,
    exclude_patterns: list[str] | None = None,
) -> dict[str, str]:
    """Read every discovered document, keyed by its POSIX path.

    All reads happen before any write, so an unreadable file aborts the run while the
    existing index and manifest are still intact.
    """

    source_texts: dict[str, str] = {}
    for path in collect_document_paths(docs_dir, exclude_patterns):
        try:
            source_texts[path.as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError(f"Could not read source document {path}: {exc}") from exc
    return source_texts


def require_docs_directory(docs_dir: Path) -> None:
    """Raise unless `docs_dir` exists and is a directory."""

    if not docs_dir.exists():
        raise ConfigurationError(
            f"DOCS_DIR does not exist: {docs_dir}",
            action_hint="Create the directory or set DOCS_DIR to an existing directory.",
        )
    if not docs_dir.is_dir():
        raise ConfigurationError(f"DOCS_DIR is not a directory: {docs_dir}")


def _is_excluded(path: Path, docs_dir: Path, exclude_patterns: list[str]) -> bool:
    # Patterns are matched against both forms so `notes/*` and `docs/notes/*` both work.
    relative_path = path.relative_to(docs_dir).as_posix()
    full_path = path.as_posix()
    return any(
        fnmatch(relative_path, pattern) or fnmatch(full_path, pattern)
        for pattern in exclude_patterns
    )
