"""Documentation must stay navigable and truthful as public surfaces change.

A link to a file that moved, a repository path quoted in prose that no longer
exists, or a document nobody can reach from the index all rot silently:
nothing fails, and readers simply stop trusting the docs. These tests turn each
of them into a failure. They also derive settings, routes, and CLI flags from
source and validate HTTP examples against the response schemas. Narrow guards
hold the root gate commands and backend tree layout where a stale instruction
would otherwise remain syntactically valid.

Archived reviews under `docs/reviews/` are dated snapshots. They describe the
repository as it was, so their links and paths are deliberately not checked.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.evals import comparison

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "README.md"
ARCHIVE_DIR = DOCS_DIR / "reviews"
REFERENCE_DIR = DOCS_DIR / "reference"
PACKAGE_DIR = REPO_ROOT / "backend" / "src" / "local_docs_rag_agent"
ROOT_AGENTS = REPO_ROOT / "AGENTS.md"
ROOT_README = REPO_ROOT / "README.md"
HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete"})

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


# --- References must cover what the code exposes ----------------------------
#
# Each surface is read from the source rather than listed here, so adding a
# setting, a route, or a flag without documenting it fails a test instead of
# leaving a reference that silently stopped being complete. Names must appear
# backticked, so `TOP_K` is not satisfied by `RERANK_TOP_K` and `POST /api/eval`
# is not satisfied by `POST /api/eval/compare`.


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _reference(name: str) -> str:
    return (REFERENCE_DIR / name).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    start = text.index(f"{heading}\n")
    level = len(heading) - len(heading.lstrip("#"))
    following_heading = re.search(
        rf"^#{{1,{level}}} ", text[start + len(heading) :], re.MULTILINE
    )
    end = (
        start + len(heading) + following_heading.start()
        if following_heading
        else len(text)
    )
    return text[start:end]


def _json_example_after(text: str, marker: str) -> object:
    remainder = text[text.index(marker) + len(marker) :]
    match = re.search(
        r"^[ \t]*```json\s*\n(.*?)\n[ \t]*```",
        remainder,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    return json.loads(match.group(1))


def _configured_variables() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_parse(PACKAGE_DIR / "config" / "__init__.py")):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
            and node.func.attr != "load_project_dotenv"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.endswith("_VARIABLES")
                for target in node.targets
            )
            and isinstance(node.value, ast.Tuple)
        ):
            names.update(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            )
    return names


def _api_routes() -> set[str]:
    tree = _parse(PACKAGE_DIR / "api" / "routes.py")
    prefix = ""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "APIRouter"
        ):
            for keyword in node.keywords:
                if keyword.arg == "prefix" and isinstance(
                    keyword.value, ast.Constant
                ):
                    prefix = str(keyword.value.value)
    routes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in HTTP_VERBS
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                verb = decorator.func.attr.upper()
                routes.add(f"{verb} {prefix}{decorator.args[0].value}")
    return routes


def _cli_surface() -> set[str]:
    names: set[str] = {axis.flag for axis in comparison.AXES}
    for node in ast.walk(_parse(PACKAGE_DIR / "cli.py")):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        first = node.args[0].value
        if node.func.attr == "add_parser" or (
            node.func.attr == "add_argument" and first.startswith("--")
        ):
            names.add(first)
    return names


def test_configuration_reference_names_every_setting() -> None:
    variables = _configured_variables()
    # A broken extractor would otherwise pass by finding nothing to check.
    assert len(variables) >= 30
    text = _reference("configuration.md")

    assert sorted(name for name in variables if f"`{name}`" not in text) == []


def test_http_api_reference_names_every_route() -> None:
    routes = _api_routes()
    assert len(routes) >= 7
    text = _reference("http-api.md")

    assert sorted(route for route in routes if f"`{route}`" not in text) == []


def test_cli_reference_names_every_command_and_flag() -> None:
    surface = _cli_surface()
    assert {"ingest", "ask", "eval", "eval-compare"} <= surface
    text = _reference("cli.md")

    assert sorted(name for name in surface if f"`{name}`" not in text) == []


def test_readme_layout_tracks_the_core_and_config_packages() -> None:
    readme = ROOT_README.read_text(encoding="utf-8")
    layout = _section(readme, "## Repository layout")

    assert "|           |-- config/" in layout
    assert "|           |-- core/" in layout
    for obsolete_entry in (
        "|           |-- config.py",
        "|           |-- constants.py",
        "|           |-- env.py",
        "|           |-- exceptions.py",
        "|           |-- models.py",
        "rag/file_io.py",
    ):
        assert obsolete_entry not in layout


def test_root_agents_lists_every_locked_quality_gate() -> None:
    source = ROOT_AGENTS.read_text(encoding="utf-8").replace("\\\n", " ")
    agents = re.sub(r"\s+", " ", source).strip()
    expected_commands = (
        "uv sync --locked --all-extras",
        "uv run --locked --all-extras python -m ruff check "
        "backend/src tests scripts",
        "uv run --locked --all-extras python -m ruff format --check "
        "backend/src tests scripts",
        "uv run --locked --all-extras python -m ruff check --preview "
        "--select DOC201 backend/src scripts",
        "uv run --locked --all-extras python -m mypy",
        "uv run --locked --all-extras python -m pytest",
        "uv run --locked --all-extras python -m compileall -q backend/src",
        "pnpm install --frozen-lockfile",
        "pnpm --filter local-docs-rag-agent-web test",
        "pnpm run build",
    )

    assert [
        command for command in expected_commands if command not in agents
    ] == []


def test_http_success_examples_match_the_response_schemas() -> None:
    text = _reference("http-api.md")
    examples = (
        ("GET /api/health", schemas.HealthResponse),
        ("GET /api/info", schemas.AppInfoResponse),
        ("GET /api/documents", schemas.DocumentsResponse),
        ("POST /api/ingest", schemas.IngestResponse),
        ("POST /api/ask", schemas.AskResponse),
        ("POST /api/eval", schemas.EvalSummaryResponse),
        ("POST /api/eval/compare", schemas.EvalCompareResponse),
    )

    for route, response_schema in examples:
        section = _section(text, f"#### `{route}`")
        payload = _json_example_after(section, "**Response**")
        assert isinstance(payload, dict), route
        assert set(payload) == set(response_schema.model_fields), route
        response_schema.model_validate(payload)

    documents_section = _section(text, "#### `GET /api/documents`")
    documents = _json_example_after(documents_section, "**Response**")
    assert isinstance(documents, dict)
    assert all(
        str(path).startswith("data/corpus/sample/")
        for path in documents["documents"]
    )


def test_http_reference_distinguishes_validation_and_domain_errors() -> None:
    error_section = _section(_reference("http-api.md"), "## Error Responses")
    validation = _json_example_after(error_section, "422 Unprocessable Entity")
    domain = _json_example_after(error_section, "Expected domain failure")

    assert isinstance(validation, dict)
    assert set(validation) == {"detail"}
    assert isinstance(validation["detail"], list)
    assert isinstance(domain, dict)
    assert set(domain) == set(schemas.ErrorResponse.model_fields)
    schemas.ErrorResponse.model_validate(domain)
