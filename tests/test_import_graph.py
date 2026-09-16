"""Import-direction contracts for parallel module work."""

from __future__ import annotations

import ast
import pathlib

PACKAGE_ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "backend"
    / "src"
    / "local_docs_rag_agent"
)

CORE_FORBIDDEN_PREFIXES = (
    "local_docs_rag_agent.config",
    "local_docs_rag_agent.rag",
    "local_docs_rag_agent.runtime",
    "local_docs_rag_agent.evals",
    "local_docs_rag_agent.providers",
    "local_docs_rag_agent.api",
    "local_docs_rag_agent.commands",
)
CORE_FORBIDDEN_NAMES = {
    "config",
    "rag",
    "runtime",
    "evals",
    "providers",
    "api",
    "commands",
    "agent",
    "tools",
    "presenters",
    "cli",
}


def _iter_python_files() -> list[pathlib.Path]:
    return [
        path for path in PACKAGE_ROOT.rglob("*.py") if path.name != "AGENTS.md"
    ]


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _is_under(path: pathlib.Path, folder: str) -> bool:
    try:
        path.relative_to(PACKAGE_ROOT / folder)
    except ValueError:
        return False
    return True


def test_core_does_not_import_other_package_modules() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        if not _is_under(path, "core"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(
                    node.module == prefix
                    or node.module.startswith(prefix + ".")
                    for prefix in CORE_FORBIDDEN_PREFIXES
                ):
                    violations.append(
                        f"{_relative(path)}:{node.lineno} imports {node.module}"
                    )
                if node.module == "local_docs_rag_agent":
                    for alias in node.names:
                        if alias.name in CORE_FORBIDDEN_NAMES:
                            violations.append(
                                f"{_relative(path)}:{node.lineno} imports "
                                f"local_docs_rag_agent.{alias.name}"
                            )
    assert violations == []


def test_production_code_outside_rag_uses_the_rag_facade() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        if _is_under(path, "rag"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if (
                node.module == "local_docs_rag_agent.rag"
                or node.module.startswith("local_docs_rag_agent.rag.")
            ):
                violations.append(
                    f"{_relative(path)}:{node.lineno} imports {node.module}"
                    f" (use local_docs_rag_agent.rag facade)"
                )
    assert violations == []
