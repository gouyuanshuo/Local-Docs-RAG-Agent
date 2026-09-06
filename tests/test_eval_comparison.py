from __future__ import annotations

import dataclasses

import pytest

from local_docs_rag_agent import cli, constants, exceptions
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.evals import comparison


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
