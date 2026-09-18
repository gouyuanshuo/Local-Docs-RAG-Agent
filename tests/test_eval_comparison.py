from __future__ import annotations

import dataclasses
import json
import pathlib
import traceback

import pytest

from local_docs_rag_agent import cli, rag
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.core import constants, exceptions, file_io, models
from local_docs_rag_agent.evals import comparison, harness

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write_eval_gold(path: pathlib.Path) -> None:
    path.write_text(
        json.dumps({"question": "What is alpha?"}), encoding="utf-8"
    )


def _comparison_config(
    tmp_path: pathlib.Path,
    *,
    vector_backend: constants.VectorBackendName = "local",
    qdrant_collection: str = "interactive",
) -> app_config.AppConfig:
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_gold(eval_path)
    return app_config.AppConfig.from_env().with_overrides(
        docs_dir=REPO_ROOT / "data" / "corpus" / "sample",
        eval_path=eval_path,
        index_path=tmp_path / "interactive.jsonl",
        ingest_manifest_path=tmp_path / "interactive-manifest.json",
        vector_backend=vector_backend,
        qdrant_url=(
            "http://qdrant.invalid:6333" if vector_backend == "qdrant" else None
        ),
        qdrant_collection=qdrant_collection,
    )


def _single_backend_request(
    backend: constants.VectorBackendName,
    *,
    chunk_sizes: list[int] | None = None,
) -> dict[str, list[object]]:
    return {
        "runtimes": ["basic"],
        "vector_backends": [backend],
        "chunk_strategies": ["markdown"],
        "retrieval_strategies": ["dense"],
        "rerankers": ["none"],
        "top_ks": [4],
        "chunk_sizes": list[object](chunk_sizes or [800]),
        "chunk_overlaps": [120],
    }


def _two_values(
    axis: comparison.MatrixAxis, config: app_config.AppConfig
) -> tuple[object, object]:
    """Return two distinct values this axis can take."""
    if axis.choices is not None:
        return axis.choices[0], axis.choices[1]
    current = getattr(config, axis.field)
    return current, current + 1


# --- Guards -----------------------------------------------------------------


def test_eval_matrix_rejects_unbounded_cartesian_product() -> None:
    with pytest.raises(exceptions.ConfigurationError, match="maximum"):
        comparison.run_eval_matrix(
            config=app_config.AppConfig.from_env(),
            requested={"top_ks": list(range(1, 130))},
        )


def test_eval_matrix_rejects_an_empty_axis() -> None:
    # An explicitly empty axis is a mistake worth reporting, not a request to
    # fall back to the default: the caller asked for nothing to be swept.
    with pytest.raises(
        exceptions.ConfigurationError, match="must not be empty"
    ):
        comparison.run_eval_matrix(
            config=app_config.AppConfig.from_env(),
            requested={"runtimes": []},
        )


def test_eval_matrix_rejects_empty_gold_before_ingest(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_path = tmp_path / "empty.jsonl"
    eval_path.write_text(" \n", encoding="utf-8")
    ingested = False

    def record_ingest(
        config: app_config.AppConfig,
    ) -> list[models.DocumentChunk]:
        del config
        nonlocal ingested
        ingested = True
        return []

    monkeypatch.setattr(rag, "ingest_documents", record_ingest)

    with pytest.raises(exceptions.DataFormatError, match="no cases"):
        comparison.run_eval_matrix(
            config=app_config.AppConfig.from_env().with_overrides(
                eval_path=eval_path
            )
        )

    assert not ingested


def test_invalid_matrix_has_no_ingest_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def record_ingest(
        config: app_config.AppConfig,
    ) -> list[models.DocumentChunk]:
        calls.append(config.chunk_size)
        return []

    monkeypatch.setattr(rag, "ingest_documents", record_ingest)
    monkeypatch.setattr(harness, "run_eval", lambda config: [])

    with pytest.raises(exceptions.ConfigurationError):
        comparison.run_eval_matrix(
            app_config.AppConfig.from_env(),
            requested={
                "vector_backends": ["local"],
                "chunk_strategies": ["markdown"],
                "retrieval_strategies": ["dense"],
                "chunk_sizes": [800, 100],
                "chunk_overlaps": [120],
            },
        )

    assert calls == []


def test_malformed_qdrant_target_fails_before_comparison_side_effects(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_user = "comparison-user-sentinel"
    sentinel_password = "comparison-password-sentinel"
    sentinel_path = "comparison-path-sentinel"
    sentinel_query = "comparison-query-sentinel"
    config = _comparison_config(tmp_path).with_overrides(
        qdrant_url=(
            f"http://{sentinel_user}:{sentinel_password}@example.invalid"
            f"\N{FULLWIDTH SOLIDUS}{sentinel_path}?key={sentinel_query}"
        )
    )
    output_path = tmp_path / "comparison-report.json"
    validated: list[constants.VectorBackendName] = []
    side_effects: list[str] = []
    validate = rag.validate_storage_target

    def record_validation(config: app_config.AppConfig) -> None:
        validated.append(config.vector_backend)
        validate(config)

    def record_side_effect(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")

    monkeypatch.setattr(rag, "validate_storage_target", record_validation)
    monkeypatch.setattr(harness, "load_eval_cases", record_side_effect)
    monkeypatch.setattr(rag, "read_source_texts", record_side_effect)
    monkeypatch.setattr(rag, "ingest_documents", record_side_effect)
    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", record_side_effect
    )
    monkeypatch.setattr(harness, "run_eval", record_side_effect)
    monkeypatch.setattr(file_io, "atomic_write_text", record_side_effect)

    with pytest.raises(exceptions.ConfigurationError) as exc_info:
        comparison.run_eval_matrix(
            config,
            {
                **_single_backend_request("local"),
                "vector_backends": ["local", "qdrant"],
            },
            output_path=output_path,
        )

    formatted = "".join(traceback.format_exception(exc_info.value))
    for sentinel in (
        sentinel_user,
        sentinel_password,
        sentinel_path,
        sentinel_query,
    ):
        assert sentinel not in str(exc_info.value)
        assert sentinel not in repr(exc_info.value)
        assert sentinel not in formatted
    assert validated == ["local", "qdrant"]
    assert side_effects == []
    assert not output_path.exists()


@pytest.mark.parametrize("separator", ["\u3002", "\uff0e", "\uff61"])
def test_idna_separator_fails_before_local_first_comparison_side_effects(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    separator: str,
) -> None:
    sentinel_user = "separator-user-sentinel"
    sentinel_password = "separator-password-sentinel"
    sentinel_path = "separator-path-sentinel"
    sentinel_query = "separator-query-sentinel"
    configured_url = (
        f"http://{sentinel_user}:{sentinel_password}@"
        f"bad{separator}host.invalid/{sentinel_path}?key={sentinel_query}"
    )
    config = _comparison_config(tmp_path).with_overrides(
        qdrant_url=configured_url
    )
    output_path = tmp_path / "comparison-report.json"
    validated: list[constants.VectorBackendName] = []
    validate = rag.validate_storage_target

    def record_validation(config: app_config.AppConfig) -> None:
        validated.append(config.vector_backend)
        validate(config)

    def unexpected_side_effect(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("comparison side effect preceded target preflight")

    monkeypatch.setattr(rag, "validate_storage_target", record_validation)
    monkeypatch.setattr(harness, "load_eval_cases", unexpected_side_effect)
    monkeypatch.setattr(rag, "read_source_texts", unexpected_side_effect)
    monkeypatch.setattr(rag, "ingest_documents", unexpected_side_effect)
    monkeypatch.setattr(
        rag,
        "initialize_owned_qdrant_index",
        unexpected_side_effect,
    )
    monkeypatch.setattr(harness, "run_eval", unexpected_side_effect)
    monkeypatch.setattr(file_io, "atomic_write_text", unexpected_side_effect)

    with pytest.raises(exceptions.ConfigurationError) as exc_info:
        comparison.run_eval_matrix(
            config,
            {
                **_single_backend_request("local"),
                "vector_backends": ["local", "qdrant"],
            },
            output_path=output_path,
        )

    rendered = "\n".join(
        (
            str(exc_info.value),
            repr(exc_info.value),
            "".join(traceback.format_exception(exc_info.value)),
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    for secret in (
        sentinel_user,
        sentinel_password,
        sentinel_path,
        sentinel_query,
        configured_url,
    ):
        assert secret not in rendered
    assert validated == ["local", "qdrant"]
    assert not output_path.exists()


def test_missing_qdrant_url_stays_a_planned_skipped_cell(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path).with_overrides(
        vector_backend="qdrant",
        qdrant_url=None,
    )

    def unexpected_validation(config: app_config.AppConfig) -> None:
        del config
        raise AssertionError("missing URL must remain skipped")

    monkeypatch.setattr(rag, "validate_storage_target", unexpected_validation)

    plan = comparison.plan_eval_matrix(
        config,
        _single_backend_request("qdrant"),
    )

    assert len(plan.variants) == 1
    assert comparison._run_matrix_case(plan.variants[0])["status"] == "skipped"


def test_normalized_qdrant_failure_does_not_leak_client_context(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_url = (
        "http://client-user-sentinel:client-password-sentinel@"
        "example.invalid/client-path-sentinel?key=client-query-sentinel"
    )
    raw_message = "client-message-sentinel"
    config = _comparison_config(tmp_path, vector_backend="qdrant")

    def normalized_failure(config: app_config.AppConfig) -> object:
        del config
        try:
            raise RuntimeError(f"{raw_message}: {raw_url}")
        except RuntimeError as exc:
            raise exceptions.VectorStoreError(
                "Qdrant operation failed",
                reason_code="operation_failed",
            ) from exc

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", normalized_failure
    )

    report = comparison.run_eval_matrix(
        config,
        _single_backend_request("qdrant"),
    )

    serialized = json.dumps(report)
    for sentinel in (
        "client-user-sentinel",
        "client-password-sentinel",
        "client-path-sentinel",
        "client-query-sentinel",
        raw_message,
    ):
        assert sentinel not in serialized
    runs = report["runs"]
    assert isinstance(runs, list)
    assert runs[0]["error"] == "VectorStoreError: Qdrant operation failed"


def test_local_matrix_cells_isolate_interactive_storage_on_success_and_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path)
    config.index_path.write_bytes(b"interactive index")
    config.ingest_manifest_path.write_bytes(b"interactive manifest")
    original_index = config.index_path.read_bytes()
    original_manifest = config.ingest_manifest_path.read_bytes()
    isolated_paths: list[tuple[pathlib.Path, pathlib.Path]] = []

    def write_isolated(
        config: app_config.AppConfig,
    ) -> list[models.DocumentChunk]:
        isolated_paths.append((config.index_path, config.ingest_manifest_path))
        config.index_path.write_text("isolated index", encoding="utf-8")
        config.ingest_manifest_path.write_text(
            "isolated manifest", encoding="utf-8"
        )
        if config.chunk_size == 900:
            raise RuntimeError("cell failure")
        return []

    monkeypatch.setattr(rag, "ingest_documents", write_isolated)
    monkeypatch.setattr(
        harness, "run_eval", lambda config: [_eval_result(_live_diagnostics())]
    )

    report = comparison.run_eval_matrix(
        config,
        _single_backend_request("local", chunk_sizes=[800, 900]),
    )

    assert config.index_path.read_bytes() == original_index
    assert config.ingest_manifest_path.read_bytes() == original_manifest
    assert len(isolated_paths) == 2
    assert all(index != config.index_path for index, _ in isolated_paths)
    assert all(
        manifest != config.ingest_manifest_path
        for _, manifest in isolated_paths
    )
    assert all(not index.exists() for index, _ in isolated_paths)
    assert all(not manifest.exists() for _, manifest in isolated_paths)
    assert isinstance(report["dataset_identity"], str)
    runs = report["runs"]
    assert isinstance(runs, list)
    assert [run["status"] for run in runs if isinstance(run, dict)] == [
        "ok",
        "error",
    ]
    identities = {
        run["configuration_identity"] for run in runs if isinstance(run, dict)
    }
    assert len(identities) == 2


def test_qdrant_matrix_cells_delete_only_owned_collections(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    created: dict[object, str] = {}
    deleted: list[str] = []

    def initialize(config: app_config.AppConfig) -> object:
        token = object()
        created[token] = config.qdrant_collection
        return token

    def delete(config: app_config.AppConfig, token: object) -> None:
        assert created[token] == config.qdrant_collection
        deleted.append(config.qdrant_collection)

    def run_eval(config: app_config.AppConfig) -> list[models.EvalResult]:
        if config.chunk_size == 900:
            raise RuntimeError("evaluation failure")
        return [_eval_result(_live_diagnostics())]

    monkeypatch.setattr(rag, "initialize_owned_qdrant_index", initialize)
    monkeypatch.setattr(rag, "delete_owned_qdrant_index", delete)
    monkeypatch.setattr(harness, "run_eval", run_eval)

    report = comparison.run_eval_matrix(
        config,
        _single_backend_request("qdrant", chunk_sizes=[800, 900]),
    )

    assert len(set(created.values())) == 2
    assert all(name != "interactive" for name in created.values())
    assert sorted(deleted) == sorted(created.values())
    assert "interactive" not in deleted
    runs = report["runs"]
    assert isinstance(runs, list)
    assert all(isinstance(run, dict) for run in runs)
    assert all(
        run["run_metadata"]["disposable_qdrant_collection"] in created.values()
        for run in runs
        if isinstance(run, dict)
    )


def test_qdrant_initializer_failure_does_not_claim_cleanup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    deleted: list[object] = []

    def initialize(config: app_config.AppConfig) -> object:
        del config
        raise RuntimeError("initializer failure")

    def delete(config: app_config.AppConfig, token: object) -> None:
        del config
        deleted.append(token)

    monkeypatch.setattr(rag, "initialize_owned_qdrant_index", initialize)
    monkeypatch.setattr(rag, "delete_owned_qdrant_index", delete)

    run = comparison._run_matrix_case(config)

    assert run["status"] == "error"
    assert deleted == []
    metadata = run["run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["disposable_qdrant_collection"] != "interactive"


def test_qdrant_initializer_cleanup_words_do_not_claim_an_orphan(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")

    def initialize(config: app_config.AppConfig) -> object:
        del config
        error = RuntimeError("initializer failure")
        error.add_note(
            "Partial Qdrant collection cleanup also failed; use exact-name "
            "orphan recovery."
        )
        raise error

    monkeypatch.setattr(rag, "initialize_owned_qdrant_index", initialize)

    run = comparison._run_matrix_case(config)

    assert run["status"] == "error"
    assert "Partial Qdrant collection cleanup also failed" in str(run["error"])
    metadata = run["run_metadata"]
    assert isinstance(metadata, dict)
    assert "orphan_recovery_required" not in metadata


def test_qdrant_initializer_structured_orphan_marker_requires_recovery(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    initialized: list[str] = []

    def initialize(config: app_config.AppConfig) -> object:
        initialized.append(config.qdrant_collection)
        raise RuntimeError("initializer failure")

    monkeypatch.setattr(rag, "initialize_owned_qdrant_index", initialize)
    monkeypatch.setattr(
        rag,
        "qdrant_orphaned_collection",
        lambda exc: initialized[0],
    )

    run = comparison._run_matrix_case(config)

    assert run["status"] == "error"
    metadata = run["run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["orphan_recovery_required"] is True
    assert metadata["disposable_qdrant_collection"] == initialized[0]


def test_qdrant_cleanup_failure_reports_exact_orphan_collection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )
    monkeypatch.setattr(
        harness, "run_eval", lambda config: [_eval_result(_live_diagnostics())]
    )

    def fail_cleanup(config: app_config.AppConfig, ownership: object) -> None:
        del config
        assert ownership is token
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(rag, "delete_owned_qdrant_index", fail_cleanup)

    run = comparison._run_matrix_case(config)

    assert run["status"] == "error"
    assert "cleanup failure" in str(run["error"])
    metadata = run["run_metadata"]
    assert isinstance(metadata, dict)
    collection = metadata["disposable_qdrant_collection"]
    assert isinstance(collection, str)
    assert collection in str(run["error"])
    assert metadata["orphan_recovery_required"] is True


def test_qdrant_interrupt_still_deletes_owned_collection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()
    deleted: list[object] = []

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )
    monkeypatch.setattr(
        harness,
        "run_eval",
        lambda config: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    monkeypatch.setattr(
        rag,
        "delete_owned_qdrant_index",
        lambda config, ownership: deleted.append(ownership),
    )

    with pytest.raises(KeyboardInterrupt, match="stop"):
        comparison._run_matrix_case(config)

    assert deleted == [token]


def test_qdrant_interrupt_preserves_active_exception_on_cleanup_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )
    monkeypatch.setattr(
        harness,
        "run_eval",
        lambda config: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    monkeypatch.setattr(
        rag,
        "delete_owned_qdrant_index",
        lambda config, ownership: (_ for _ in ()).throw(SystemExit("cleanup")),
    )

    with pytest.raises(KeyboardInterrupt, match="stop") as raised:
        comparison._run_matrix_case(config)

    notes = getattr(raised.value, "__notes__", [])
    assert any("disposable collection" in note for note in notes)
    assert any("SystemExit" in note for note in notes)


def test_qdrant_cleanup_interrupt_after_completed_eval_has_recovery_note(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )
    monkeypatch.setattr(
        harness,
        "run_eval",
        lambda config: [_eval_result(_live_diagnostics())],
    )
    monkeypatch.setattr(
        rag,
        "delete_owned_qdrant_index",
        lambda config, ownership: (_ for _ in ()).throw(SystemExit("cleanup")),
    )

    with pytest.raises(SystemExit, match="cleanup") as raised:
        comparison._run_matrix_case(config)

    notes = getattr(raised.value, "__notes__", [])
    assert any("disposable collection" in note for note in notes)
    assert any("exact-name orphan recovery" in note for note in notes)


def test_qdrant_cleanup_interrupt_after_cell_error_has_recovery_note(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )
    monkeypatch.setattr(
        harness,
        "run_eval",
        lambda config: (_ for _ in ()).throw(RuntimeError("eval failure")),
    )
    monkeypatch.setattr(
        rag,
        "delete_owned_qdrant_index",
        lambda config, ownership: (_ for _ in ()).throw(
            KeyboardInterrupt("cleanup")
        ),
    )

    with pytest.raises(KeyboardInterrupt, match="cleanup") as raised:
        comparison._run_matrix_case(config)

    notes = getattr(raised.value, "__notes__", [])
    assert any("disposable collection" in note for note in notes)
    assert any("exact-name orphan recovery" in note for note in notes)


@pytest.mark.parametrize("outcome", ["ok", "degraded", "skipped", "error"])
def test_qdrant_cleanup_failure_preserves_cell_outcome(
    outcome: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _comparison_config(tmp_path, vector_backend="qdrant")
    token = object()

    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", lambda config: token
    )

    if outcome == "ok":
        monkeypatch.setattr(
            harness,
            "run_eval",
            lambda config: [_eval_result(_live_diagnostics())],
        )
    elif outcome == "degraded":
        monkeypatch.setattr(
            harness,
            "run_eval",
            lambda config: [
                _eval_result(_live_diagnostics(chat_mode="fallback"))
            ],
        )
    elif outcome == "skipped":
        monkeypatch.setattr(
            harness,
            "run_eval",
            lambda config: (_ for _ in ()).throw(
                exceptions.VectorStoreError(
                    "Qdrant unreachable", reason_code="unreachable"
                )
            ),
        )
    else:
        monkeypatch.setattr(
            harness,
            "run_eval",
            lambda config: (_ for _ in ()).throw(RuntimeError("eval failure")),
        )

    def fail_cleanup(config: app_config.AppConfig, ownership: object) -> None:
        del config
        assert ownership is token
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(rag, "delete_owned_qdrant_index", fail_cleanup)

    run = comparison._run_matrix_case(config)

    assert run["status"] == "error"
    assert comparison._build_leaderboard([run]) == []
    metadata = run["run_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["pre_cleanup_status"] == outcome
    assert metadata["orphan_recovery_required"] is True
    assert metadata["cleanup_error"] == "RuntimeError: cleanup failure"
    if outcome in {"ok", "degraded"}:
        assert isinstance(run["summary"], dict)
    if outcome in {"degraded", "skipped"}:
        assert run["reason"]
    if outcome == "error":
        assert metadata["prior_error"] == "RuntimeError: eval failure"


def test_comparison_identities_ignore_isolated_paths_and_change_meaningfully(
    tmp_path: pathlib.Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    document = docs_dir / "source.md"
    document.write_text("first corpus", encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_gold(eval_path)
    base = app_config.AppConfig.from_env().with_overrides(
        docs_dir=docs_dir,
        eval_path=eval_path,
        index_path=tmp_path / "interactive-a.jsonl",
        ingest_manifest_path=tmp_path / "interactive-a.json",
    )
    isolated = base.with_overrides(
        index_path=tmp_path / "other" / "index.jsonl",
        ingest_manifest_path=tmp_path / "other" / "manifest.json",
    )

    dataset_before = comparison._dataset_identity(base)
    configuration_before = comparison._configuration_identity(base)
    assert dataset_before == comparison._dataset_identity(isolated)
    assert configuration_before == (
        comparison._configuration_identity(isolated)
    )

    document.write_text("changed corpus", encoding="utf-8")
    assert dataset_before != comparison._dataset_identity(base)
    assert configuration_before != (
        comparison._configuration_identity(base.with_overrides(chunk_size=900))
    )

    copied_docs_dir = tmp_path / "copied-docs"
    copied_docs_dir.mkdir()
    (copied_docs_dir / "source.md").write_text(
        "changed corpus", encoding="utf-8"
    )
    copied_eval_path = tmp_path / "copied-eval.jsonl"
    copied_eval_path.write_bytes(eval_path.read_bytes())
    copied = base.with_overrides(
        docs_dir=copied_docs_dir,
        eval_path=copied_eval_path,
        index_path=tmp_path / "copied-index.jsonl",
        ingest_manifest_path=tmp_path / "copied-manifest.json",
        vector_backend="qdrant",
        qdrant_url="http://qdrant.invalid:6333",
        qdrant_collection="interactive-eval-0123456789abcdef",
    )
    qdrant_variant = copied.with_overrides(
        qdrant_collection="interactive-eval-fedcba9876543210"
    )

    assert comparison._dataset_identity(base) == comparison._dataset_identity(
        copied
    )
    assert comparison._configuration_identity(copied) == (
        comparison._configuration_identity(qdrant_variant)
    )


def test_dataset_identity_keeps_lexical_symlink_aliases(
    tmp_path: pathlib.Path,
) -> None:
    def make_corpus(root: pathlib.Path, alias: str) -> pathlib.Path:
        root.mkdir()
        (root / "target.md").write_text("same corpus", encoding="utf-8")
        (root / alias).symlink_to("target.md")
        return root

    eval_path = tmp_path / "eval.jsonl"
    _write_eval_gold(eval_path)
    first_docs = make_corpus(tmp_path / "first-docs", "alias.md")
    copied_docs = make_corpus(tmp_path / "copied-docs", "alias.md")
    renamed_docs = make_corpus(tmp_path / "renamed-docs", "renamed.md")
    base = app_config.AppConfig.from_env().with_overrides(eval_path=eval_path)

    first = base.with_overrides(docs_dir=first_docs)
    copied = base.with_overrides(docs_dir=copied_docs)
    renamed = base.with_overrides(docs_dir=renamed_docs)

    assert comparison._dataset_identity(first) == comparison._dataset_identity(
        copied
    )
    assert comparison._dataset_identity(first) != comparison._dataset_identity(
        renamed
    )


def test_matrix_plan_reports_the_twelve_default_local_cells() -> None:
    plan = comparison.plan_eval_matrix(app_config.AppConfig.from_env())

    assert plan.axes["vector_backends"] == ["local"]
    assert plan.axes["chunk_strategies"] == ["fixed", "paragraph", "markdown"]
    assert plan.axes["retrieval_strategies"] == [
        "blended",
        "dense",
        "lexical",
        "hybrid_rrf",
    ]
    assert plan.num_runs == 12
    assert len(plan.variants) == 12


def test_matrix_plan_reports_twenty_four_cells_with_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")

    plan = comparison.plan_eval_matrix(app_config.AppConfig.from_env())

    assert plan.axes["vector_backends"] == ["local", "qdrant"]
    assert plan.num_runs == 24
    assert len(plan.variants) == 24


def test_matrix_execution_uses_the_planned_axes_and_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = app_config.AppConfig.from_env()
    requested: dict[str, list[object]] = {
        "runtimes": ["basic"],
        "vector_backends": ["local"],
        "chunk_strategies": ["markdown"],
        "retrieval_strategies": ["dense"],
        "rerankers": ["none"],
        "top_ks": [4, 5],
        "chunk_sizes": [800],
        "chunk_overlaps": [120],
    }
    plan = comparison.plan_eval_matrix(config, requested)
    executed: list[app_config.AppConfig] = []

    def record_run(
        config: app_config.AppConfig,
        *,
        dataset_identity: str | None = None,
    ) -> dict[str, object]:
        del dataset_identity
        executed.append(config)
        return {
            "label": comparison.run_label(config),
            "status": "skipped",
            "reason": "test",
        }

    monkeypatch.setattr(comparison, "_run_matrix_case", record_run)

    report = comparison.run_eval_matrix(config, requested)

    assert report["num_runs"] == plan.num_runs == 2
    assert {axis.name: report[axis.name] for axis in comparison.AXES} == (
        plan.axes
    )
    assert executed == plan.variants


# --- The axis table is the single definition ---------------------------------


def test_every_axis_overrides_a_real_configuration_field() -> None:
    fields = {field.name for field in dataclasses.fields(app_config.AppConfig)}

    assert {axis.field for axis in comparison.AXES} <= fields


def test_request_schema_covers_every_matrix_axis() -> None:
    # `compare_eval` hands the request straight to `run_eval_matrix` keyed by
    # axis name, so a field renamed on one side and not the other would
    # silently stop sweeping that axis.
    assert {axis.name for axis in comparison.AXES} == set(
        schemas.EvalCompareRequest.model_fields
    )


def test_response_schema_reports_every_matrix_axis() -> None:
    assert {axis.name for axis in comparison.AXES} <= set(
        schemas.EvalCompareResponse.model_fields
    )


def test_cli_exposes_a_flag_for_every_axis() -> None:
    parser = cli.build_parser()
    argv = ["eval-compare"]
    for axis in comparison.AXES:
        value = axis.choices[0] if axis.choices is not None else "2"
        argv += [axis.flag, str(value)]

    args = parser.parse_args(argv)

    for axis in comparison.AXES:
        assert getattr(args, axis.name), axis.name


@pytest.mark.parametrize(
    "axis", comparison.AXES, ids=lambda axis: str(axis.name)
)
def test_run_label_distinguishes_every_axis(
    axis: comparison.MatrixAxis,
) -> None:
    # Two cells that differ in any axis must not share a label, or the
    # leaderboard reports one of them twice and loses the other. Deriving the
    # label from the table is what keeps this true as axes are added.
    config = app_config.AppConfig.from_env()
    first, second = _two_values(axis, config)

    left = comparison.run_label(config.with_overrides(**{axis.field: first}))
    right = comparison.run_label(config.with_overrides(**{axis.field: second}))

    assert left != right


# --- Defaults ----------------------------------------------------------------


def test_omitted_axes_take_their_defaults() -> None:
    config = app_config.AppConfig.from_env()

    axes = comparison.resolve_axes(config, {"top_ks": [1, 2]})

    assert axes["top_ks"] == [1, 2]
    assert axes["retrieval_strategies"] == list(constants.RETRIEVAL_STRATEGIES)
    assert axes["chunk_strategies"] == list(constants.CHUNK_STRATEGIES)
    # Reranking is the one axis that holds steady, because every `llm` cell
    # spends a model call per eval case.
    assert axes["rerankers"] == [config.reranker]
    assert axes["chunk_sizes"] == [config.chunk_size]


def test_qdrant_is_only_compared_when_a_url_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = app_config.AppConfig.from_env()
    assert comparison.resolve_axes(config, {})["vector_backends"] == ["local"]

    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid:6333")
    with_url = app_config.AppConfig.from_env()

    assert comparison.resolve_axes(with_url, {})["vector_backends"] == [
        "local",
        "qdrant",
    ]


def _eval_result(
    diagnostics: models.AnswerDiagnostics | None,
) -> models.EvalResult:
    return models.EvalResult(
        question="q",
        answer="a",
        citations=[],
        retrieved_sources=[],
        answer_keyword_hit_rate=1.0,
        retrieval_source_hit_rate=1.0,
        retrieval_span_hit_rate=1.0,
        retrieval_reciprocal_rank=1.0,
        retrieval_precision=1.0,
        citation_source_hit_rate=1.0,
        citation_span_hit_rate=1.0,
        response_time_ms=1.0,
        diagnostics=diagnostics,
    )


def _live_diagnostics(
    *,
    requested: constants.RuntimeName = "basic",
    actual: constants.RuntimeName = "basic",
    chat_mode: models.ProviderMode = "live",
    embedding_mode: models.ProviderMode = "live",
    reranker_mode: models.ProviderMode = "ready",
) -> models.AnswerDiagnostics:
    return models.AnswerDiagnostics(
        requested_runtime=requested,
        actual_runtime=actual,
        vector_backend="local",
        chat_provider=models.ProviderStatus(provider="chat", mode=chat_mode),
        embedding_provider=models.ProviderStatus(
            provider="embedding", mode=embedding_mode
        ),
        reranker=models.ProviderStatus(
            provider="none", mode=reranker_mode, reason="reranker_disabled"
        ),
    )


def _ok_summary(
    *, reciprocal_rank: float, span_hit_rate: float
) -> dict[str, object]:
    return {
        "retrieval_reciprocal_rank": reciprocal_rank,
        "retrieval_precision": 0.5,
        "retrieval_span_hit_rate": span_hit_rate,
        "answer_keyword_hit_rate": 0.5,
        "citation_span_hit_rate": 0.5,
        "avg_response_time_ms": 10.0,
    }


def test_leaderboard_ranks_reciprocal_rank_ahead_of_span_hit_rate() -> None:
    # If the sort key regresses to retrieval_span_hit_rate, the high-span
    # low-RR row would win. That is the blindness the rank-aware metrics
    # exist to prevent.
    runs: list[dict[str, object]] = [
        {
            "label": "low-rr-high-span",
            "status": "ok",
            "summary": _ok_summary(reciprocal_rank=0.25, span_hit_rate=1.0),
        },
        {
            "label": "high-rr-low-span",
            "status": "ok",
            "summary": _ok_summary(reciprocal_rank=1.0, span_hit_rate=0.5),
        },
        {
            "label": "degraded-high-rr",
            "status": "degraded",
            "reason": "chat_fallback",
            "summary": _ok_summary(reciprocal_rank=1.0, span_hit_rate=1.0),
        },
    ]

    board = comparison._build_leaderboard(runs)

    assert [row["label"] for row in board] == [
        "high-rr-low-span",
        "low-rr-high-span",
    ]


def test_zero_results_are_not_rankable() -> None:
    assert comparison.cell_degradation_reason([]) == "no_eval_cases"


def test_missing_diagnostics_are_not_rankable() -> None:
    assert comparison.cell_degradation_reason([_eval_result(None)]) == (
        "missing_diagnostics"
    )


def test_fallback_cell_is_degraded_and_absent_from_leaderboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "ingest_documents", lambda config: [])
    fallback = _eval_result(
        _live_diagnostics(chat_mode="fallback", embedding_mode="fallback")
    )
    monkeypatch.setattr(harness, "run_eval", lambda config: [fallback])

    run = comparison._run_matrix_case(app_config.AppConfig.from_env())

    assert run["status"] == "degraded"
    assert "chat_fallback" in str(run["reason"])
    assert "embedding_fallback" in str(run["reason"])
    assert comparison._build_leaderboard([run]) == []


def test_runtime_mismatch_is_degraded_not_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "ingest_documents", lambda config: [])
    mismatch = _eval_result(
        _live_diagnostics(requested="agents_sdk", actual="basic")
    )
    monkeypatch.setattr(harness, "run_eval", lambda config: [mismatch])

    run = comparison._run_matrix_case(app_config.AppConfig.from_env())

    assert run["status"] == "degraded"
    assert run["reason"] == "runtime_fallback"
    assert comparison._build_leaderboard([run]) == []


def test_unknown_search_diagnostics_are_degraded_and_not_ranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "ingest_documents", lambda config: [])
    no_search = _eval_result(
        models.AnswerDiagnostics(
            requested_runtime="agents_sdk",
            actual_runtime="agents_sdk",
            vector_backend="local",
            chat_provider=models.ProviderStatus(provider="chat", mode="live"),
            embedding_provider=models.ProviderStatus(
                provider="embedding",
                mode="unknown",
                reason="search_not_run",
            ),
            reranker=models.ProviderStatus(
                provider="reranker",
                mode="unknown",
                reason="search_not_run",
            ),
        )
    )
    monkeypatch.setattr(harness, "run_eval", lambda config: [no_search])

    run = comparison._run_matrix_case(
        app_config.AppConfig.from_env().with_overrides(
            agent_runtime="agents_sdk"
        )
    )

    assert run["status"] == "degraded"
    assert run["reason"] == "embedding_not_live,reranker_not_live"
    assert comparison._build_leaderboard([run]) == []


def test_live_cell_with_disabled_reranker_stays_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "ingest_documents", lambda config: [])
    live = _eval_result(_live_diagnostics())
    monkeypatch.setattr(harness, "run_eval", lambda config: [live])

    run = comparison._run_matrix_case(app_config.AppConfig.from_env())

    assert run["status"] == "ok"
    assert comparison._build_leaderboard([run])[0]["label"] == run["label"]


def test_no_key_compare_has_no_leaderboard_winner(
    tmp_path: pathlib.Path,
) -> None:
    config = app_config.AppConfig.from_env().with_overrides(
        docs_dir=REPO_ROOT / "data" / "corpus" / "sample",
        eval_path=REPO_ROOT / "data" / "evals" / "sample_eval.jsonl",
        index_path=tmp_path / "chunks.jsonl",
        ingest_manifest_path=tmp_path / "manifest.json",
        vector_backend="local",
        embedding_api_key=None,
        llm_api_key=None,
    )
    report = comparison.run_eval_matrix(
        config,
        requested={
            "runtimes": ["basic"],
            "chunk_strategies": ["markdown"],
            "retrieval_strategies": ["lexical"],
            "vector_backends": ["local"],
            "rerankers": ["none"],
            "top_ks": [4],
            "chunk_sizes": [800],
            "chunk_overlaps": [120],
        },
    )

    assert report["leaderboard"] == []
    runs = report["runs"]
    assert isinstance(runs, list)
    assert runs
    assert all(isinstance(run, dict) and run["status"] != "ok" for run in runs)
    assert all(
        isinstance(run, dict) and run["status"] == "degraded" for run in runs
    )
