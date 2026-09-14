"""Documentation must stay navigable as the code and the tree change.

A link to a file that moved, a repository path quoted in prose that no longer
exists, or a document nobody can reach from the index all rot silently:
nothing fails, and readers simply stop trusting the docs. These tests turn each
of them into a failure.

Archived reviews under `docs/reviews/` are dated snapshots. They describe the
repository as it was, so their links and paths are deliberately not checked.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "README.md"
ARCHIVE_DIR = DOCS_DIR / "reviews"

FENCE_RE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:")

# A quoted repository path: a root document, or a path under a top-level
# directory. Globs and placeholders (`*`, `<x>`) never match, so they are left
# alone rather than reported.
REPO_PATH_RE = re.compile(
    r"^(?:AGENTS\.md|README\.md|spec\.md|tasks\.md"
    r"|(?:docs|skills|tests|scripts|backend|frontend|data)/[\w./-]+)$"
)
# Generated or gitignored locations: quoted in prose, absent on a fresh clone.
GENERATED_PREFIXES = (
    "data/index",
    "data/evals/compare_latest.json",
    "data/evals/local_vs_qdrant_compare.json",
    "frontend/dist",
    "frontend/node_modules",
)


def _documents() -> list[pathlib.Path]:
    candidates = [
        *REPO_ROOT.glob("*.md"),
        *DOCS_DIR.rglob("*.md"),
        *(REPO_ROOT / "backend" / "src").rglob("AGENTS.md"),
        *(REPO_ROOT / "frontend").glob("*.md"),
        *(REPO_ROOT / "skills").rglob("*.md"),
    ]
    return sorted(
        {path for path in candidates if ARCHIVE_DIR not in path.parents}
    )


def _without_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def _link_target(source: pathlib.Path, target: str) -> pathlib.Path | None:
    """Resolve a markdown link target, or None when it is not a local file."""
    if target.startswith("#") or SCHEME_RE.match(target):
        return None
    path_part = target.split("#", 1)[0]
    if not path_part:
        return None
    return (source.parent / path_part).resolve()


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_relative_markdown_links_point_at_existing_files() -> None:
    broken: list[str] = []
    for source in _documents():
        text = INLINE_CODE_RE.sub(
            "", _without_fences(source.read_text(encoding="utf-8"))
        )
        for target in LINK_RE.findall(text):
            resolved = _link_target(source, target)
            if resolved is not None and not resolved.exists():
                broken.append(f"{_relative(source)} -> {target}")

    assert broken == []


def test_quoted_repository_paths_exist() -> None:
    # This repository names files in backticks far more often than it links
    # them, so a moved document leaves stale backtick paths behind long before
    # it leaves a broken link.
    stale: list[str] = []
    for source in _documents():
        text = _without_fences(source.read_text(encoding="utf-8"))
        for quoted in INLINE_CODE_RE.findall(text):
            candidate = quoted.strip().rstrip("/")
            if not REPO_PATH_RE.match(candidate):
                continue
            if candidate.startswith(GENERATED_PREFIXES):
                continue
            if not (REPO_ROOT / candidate).exists():
                stale.append(f"{_relative(source)} -> {candidate}")

    assert stale == []


def test_every_document_is_reachable_from_the_index() -> None:
    index_text = INLINE_CODE_RE.sub(
        "", _without_fences(INDEX.read_text(encoding="utf-8"))
    )
    linked = {
        _link_target(INDEX, target) for target in LINK_RE.findall(index_text)
    }

    orphans = [
        _relative(path)
        for path in sorted(DOCS_DIR.rglob("*.md"))
        if path != INDEX
        and ARCHIVE_DIR not in path.parents
        and path.resolve() not in linked
    ]

    assert orphans == []
