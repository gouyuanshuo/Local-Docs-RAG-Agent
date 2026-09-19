"""Import-direction contracts for parallel module work."""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_ROOT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "backend"
    / "src"
    / "local_docs_rag_agent"
)
PACKAGE_NAME = "local_docs_rag_agent"
RAG_FACADE = f"{PACKAGE_NAME}.rag"

DIRECTORY_LAYERS = {
    "api": "delivery",
    "commands": "delivery",
    "config": "config",
    "core": "core",
    "evals": "evals",
    "providers": "providers",
    "rag": "rag",
    "runtime": "runtime",
}
TOP_LEVEL_FILE_LAYERS = {
    "__init__.py": "facade",
    "agent.py": "runtime",
    "cli.py": "delivery",
    "presenters.py": "delivery",
    "tools.py": "runtime",
}
ALLOWED_LAYER_IMPORTS = {
    "delivery": {
        "config",
        "core",
        "delivery",
        "evals",
        "providers",
        "rag",
        "runtime",
    },
    "evals": {"config", "core", "evals", "rag", "runtime"},
    "runtime": {"config", "core", "providers", "rag", "runtime"},
    "rag": {"config", "core", "providers", "rag"},
    "providers": {"config", "core", "providers"},
    "config": {"config", "core"},
    "core": {"core"},
}
DELIVERY_FRAMEWORKS = {"argparse", "fastapi", "pydantic"}
EVALS_PRESENTER_SEAM = f"{PACKAGE_NAME}.presenters"
RAG_INTERNAL_NAMES = {
    path.stem
    for path in (PACKAGE_ROOT / "rag").glob("*.py")
    if path.name != "__init__.py"
}


def _iter_python_files() -> list[pathlib.Path]:
    return [
        path for path in PACKAGE_ROOT.rglob("*.py") if path.name != "AGENTS.md"
    ]


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _classify_file(path: pathlib.Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    if len(relative.parts) == 1:
        layer = TOP_LEVEL_FILE_LAYERS.get(relative.name)
    else:
        layer = DIRECTORY_LAYERS.get(relative.parts[0])
    if layer is None:
        raise ValueError(f"unclassified production file: {relative.as_posix()}")
    return layer


def _module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE_NAME, *parts])


def _resolve_from_module(path: pathlib.Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    module_parts = _module_name(path).split(".")
    if path.name != "__init__.py":
        module_parts.pop()
    ascents = node.level - 1
    if ascents > len(module_parts):
        return node.module or ""
    base_parts = module_parts[: len(module_parts) - ascents]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _import_targets(path: pathlib.Path, tree: ast.AST) -> list[tuple[str, int]]:
    targets: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend((alias.name, node.lineno) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        module = _resolve_from_module(path, node)
        if module == PACKAGE_NAME:
            targets.extend(
                (f"{module}.{alias.name}", node.lineno)
                for alias in node.names
                if alias.name != "*"
            )
            continue
        if module == RAG_FACADE:
            internal_names = [
                alias.name
                for alias in node.names
                if alias.name in RAG_INTERNAL_NAMES
            ]
            if internal_names:
                targets.extend(
                    (f"{module}.{name}", node.lineno) for name in internal_names
                )
                continue
        if module:
            targets.append((module, node.lineno))
    return targets


def _layer_for_module(module: str) -> str | None:
    prefix = f"{PACKAGE_NAME}."
    if not module.startswith(prefix):
        return None
    top_level_name = module.removeprefix(prefix).split(".", maxsplit=1)[0]
    if top_level_name in DIRECTORY_LAYERS:
        return DIRECTORY_LAYERS[top_level_name]
    top_level_path = f"{top_level_name}.py"
    return TOP_LEVEL_FILE_LAYERS.get(top_level_path)


def _find_import_violations(path: pathlib.Path, source: str) -> list[str]:
    source_layer = _classify_file(path)
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for module, lineno in _import_targets(path, tree):
        imported_root = module.split(".", maxsplit=1)[0]
        location = f"{_relative(path)}:{lineno}"
        if source_layer != "delivery" and imported_root in DELIVERY_FRAMEWORKS:
            violations.append(
                f"{location} {source_layer} cannot import delivery "
                f"framework {imported_root}"
            )
            continue

        target_layer = _layer_for_module(module)
        if target_layer is None:
            continue
        if source_layer != "rag" and module.startswith(f"{RAG_FACADE}."):
            violations.append(
                f"{location} imports {module}; modules outside rag must use "
                f"the {RAG_FACADE} facade"
            )
            continue
        if source_layer == "facade":
            continue
        if source_layer == "evals" and module == EVALS_PRESENTER_SEAM:
            continue
        if target_layer not in ALLOWED_LAYER_IMPORTS[source_layer]:
            violations.append(
                f"{location} {source_layer} cannot import {target_layer} "
                f"({module})"
            )
    return violations


def _synthetic_violations(relative_path: str, source: str) -> list[str]:
    return _find_import_violations(PACKAGE_ROOT / relative_path, source)


@pytest.mark.parametrize(
    ("relative_path", "source", "expected"),
    [
        pytest.param(
            "core/example.py",
            "import local_docs_rag_agent.rag\n",
            "core cannot import rag",
            id="core-to-rag",
        ),
        pytest.param(
            "providers/example.py",
            "from local_docs_rag_agent import runtime\n",
            "providers cannot import runtime",
            id="providers-to-runtime",
        ),
        pytest.param(
            "rag/example.py",
            "from ..api import schemas\n",
            "rag cannot import delivery",
            id="rag-to-api-relative",
        ),
        pytest.param(
            "runtime/example.py",
            "import fastapi\n",
            "runtime cannot import delivery framework fastapi",
            id="runtime-to-fastapi",
        ),
        pytest.param(
            "runtime/example.py",
            "from pydantic import BaseModel\n",
            "runtime cannot import delivery framework pydantic",
            id="runtime-to-pydantic",
        ),
        pytest.param(
            "providers/example.py",
            "import argparse\n",
            "providers cannot import delivery framework argparse",
            id="providers-to-argparse",
        ),
        pytest.param(
            "runtime/example.py",
            ("from local_docs_rag_agent.rag.pipeline import retrieve\n"),
            "must use the local_docs_rag_agent.rag facade",
            id="rag-internal-from-import",
        ),
        pytest.param(
            "runtime/example.py",
            "import local_docs_rag_agent.rag.pipeline\n",
            "must use the local_docs_rag_agent.rag facade",
            id="rag-internal-plain-import",
        ),
        pytest.param(
            "runtime/example.py",
            "from ..rag.pipeline import retrieve\n",
            "must use the local_docs_rag_agent.rag facade",
            id="rag-internal-relative-import",
        ),
        pytest.param(
            "runtime/example.py",
            "from local_docs_rag_agent.rag import pipeline\n",
            "must use the local_docs_rag_agent.rag facade",
            id="rag-internal-package-import",
        ),
        pytest.param(
            "evals/example.py",
            "from ..api import schemas\n",
            "evals cannot import delivery",
            id="evals-to-api",
        ),
        pytest.param(
            "evals/example.py",
            "from ..providers import factory\n",
            "evals cannot import providers",
            id="evals-to-providers",
        ),
        pytest.param(
            "runtime/example.py",
            "from ..evals import harness\n",
            "runtime cannot import evals",
            id="runtime-to-evals",
        ),
        pytest.param(
            "rag/example.py",
            "from ..runtime import dispatch\n",
            "rag cannot import runtime",
            id="rag-to-runtime",
        ),
        pytest.param(
            "providers/example.py",
            "from ..rag import retrieve\n",
            "providers cannot import rag",
            id="providers-to-rag",
        ),
        pytest.param(
            "config/example.py",
            "from ..providers import factory\n",
            "config cannot import providers",
            id="config-to-providers",
        ),
    ],
)
def test_checker_rejects_forbidden_imports(
    relative_path: str, source: str, expected: str
) -> None:
    violations = _synthetic_violations(relative_path, source)

    assert len(violations) == 1
    assert expected in violations[0]


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        pytest.param("api/routes.py", "delivery", id="api"),
        pytest.param("commands/ask.py", "delivery", id="commands"),
        pytest.param("cli.py", "delivery", id="cli"),
        pytest.param("presenters.py", "delivery", id="presenters"),
        pytest.param("evals/harness.py", "evals", id="evals"),
        pytest.param("runtime/basic.py", "runtime", id="runtime"),
        pytest.param("agent.py", "runtime", id="agent-facade"),
        pytest.param("tools.py", "runtime", id="tools-facade"),
        pytest.param("rag/pipeline.py", "rag", id="rag"),
        pytest.param("providers/factory.py", "providers", id="providers"),
        pytest.param("config/__init__.py", "config", id="config"),
        pytest.param("core/models.py", "core", id="core"),
        pytest.param("__init__.py", "facade", id="root-facade-exemption"),
    ],
)
def test_classifier_assigns_documented_layers(
    relative_path: str, expected: str
) -> None:
    assert _classify_file(PACKAGE_ROOT / relative_path) == expected


def test_classifier_rejects_an_unmapped_production_file() -> None:
    with pytest.raises(ValueError, match="unclassified production file"):
        _classify_file(PACKAGE_ROOT / "unknown.py")


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        pytest.param(
            "api/example.py",
            "\n".join(
                [
                    "import fastapi",
                    "from local_docs_rag_agent import agent, config, rag",
                    "from local_docs_rag_agent.api import schemas",
                    "from local_docs_rag_agent.evals import harness",
                    "from local_docs_rag_agent.providers import factory",
                ]
            ),
            id="delivery",
        ),
        pytest.param(
            "evals/example.py",
            "\n".join(
                [
                    "from local_docs_rag_agent import agent, config, rag",
                    "from local_docs_rag_agent import presenters",
                    "from local_docs_rag_agent.core import models",
                    "from . import harness",
                ]
            ),
            id="evals",
        ),
        pytest.param(
            "runtime/example.py",
            "\n".join(
                [
                    "import local_docs_rag_agent.rag",
                    "from local_docs_rag_agent import config, tools",
                    "from local_docs_rag_agent.core import models",
                    "from local_docs_rag_agent.providers import factory",
                    "from local_docs_rag_agent.rag import retrieve",
                    "from . import shared",
                    "from .. import rag",
                    "from ..rag import retrieve",
                ]
            ),
            id="runtime",
        ),
        pytest.param(
            "rag/example.py",
            "\n".join(
                [
                    "from local_docs_rag_agent import config",
                    "from local_docs_rag_agent.core import models",
                    "from local_docs_rag_agent.providers import base",
                    "from . import retrieval",
                ]
            ),
            id="rag",
        ),
        pytest.param(
            "providers/example.py",
            "\n".join(
                [
                    "from local_docs_rag_agent import config",
                    "from local_docs_rag_agent.core import models",
                    "from . import base",
                ]
            ),
            id="providers",
        ),
        pytest.param(
            "config/example.py",
            "\n".join(
                [
                    "from local_docs_rag_agent.core import env",
                    "from . import defaults",
                ]
            ),
            id="config",
        ),
        pytest.param(
            "core/example.py",
            "from local_docs_rag_agent.core import models\n",
            id="core",
        ),
    ],
)
def test_checker_allows_documented_dependencies(
    relative_path: str, source: str
) -> None:
    assert _synthetic_violations(relative_path, source) == []


def test_checker_allows_evals_to_use_the_presenters_shared_seam() -> None:
    source = "from local_docs_rag_agent import presenters\n"

    assert _synthetic_violations("evals/example.py", source) == []


def test_every_production_file_has_a_layer_or_facade_exemption() -> None:
    classifications = {
        _relative(path): _classify_file(path) for path in _iter_python_files()
    }

    assert set(classifications.values()) == {
        *ALLOWED_LAYER_IMPORTS,
        "facade",
    }
    assert [
        path for path, layer in classifications.items() if layer == "facade"
    ] == ["__init__.py"]


def test_production_imports_follow_documented_dependency_direction() -> None:
    violations = [
        violation
        for path in _iter_python_files()
        for violation in _find_import_violations(
            path, path.read_text(encoding="utf-8")
        )
    ]

    assert violations == []


def test_generic_ingest_lifecycle_uses_store_capabilities() -> None:
    path = PACKAGE_ROOT / "rag" / "ingest.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lifecycle_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_ingest_documents_locked", "_ensure_index_locked"}
    }
    violations: list[str] = []
    for function_name, function in lifecycle_functions.items():
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and node.attr in {
                "vector_backend",
                "collection_exists",
                "QdrantChunkStore",
            }:
                violations.append(
                    f"{function_name}:{node.lineno} uses {node.attr}"
                )
            if isinstance(node, ast.Name) and node.id == "isinstance":
                violations.append(
                    f"{function_name}:{node.lineno} uses isinstance"
                )

    assert set(lifecycle_functions) == {
        "_ingest_documents_locked",
        "_ensure_index_locked",
    }
    assert violations == []
