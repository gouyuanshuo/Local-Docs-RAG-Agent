from __future__ import annotations

import dataclasses
import pathlib

import pytest

from local_docs_rag_agent import cli, rag
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.core import constants, exceptions, models
from local_docs_rag_agent.evals import comparison, harness

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def _eval_result(diagnostics: models.AnswerDiagnostics) -> models.EvalResult:
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


def test_live_cell_stays_ok(monkeypatch: pytest.MonkeyPatch) -> None:
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
