"""Atomic text-file replacement for the index and the manifest.

Both files are rewritten in full on every ingest. Writing to a sibling temporary
file and replacing the target means an interrupted run leaves the previous
version intact rather than a truncated one that would fail to parse on the next
read.
"""

from __future__ import annotations

import os
import pathlib
import tempfile


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Write text through a sibling temporary file, then atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
