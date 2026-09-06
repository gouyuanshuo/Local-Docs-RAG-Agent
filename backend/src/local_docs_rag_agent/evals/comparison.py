"""Runs the eval harness across a matrix of retrieval configurations.

Each cell re-ingests and re-evaluates under one `AppConfig` variant so the
results are comparable. A cell that cannot run is reported as `skipped` with a
reason, and a cell that fails is reported as `error`; neither is silently
dropped, because a leaderboard that hides its gaps is worse than no leaderboard.

`AXES` is the single definition of what a comparison can sweep. The Cartesian
product, the per-cell configuration overrides, the run label, and the report
keys are all derived from it, and the CLI builds its flags from it too. That
matters because the alternative — a parameter list, a length list, a
`product()` call, and an unpacking tuple that must agree by position — fails
silently when they drift: every axis is a sequence, so binding `chunk_size` to
the reranker's values type-checks and runs, and only the numbers come out
wrong.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib
from collections.abc import Callable, Mapping, Sequence

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import constants, exceptions, presenters, rag
from local_docs_rag_agent.evals import harness
from local_docs_rag_agent.rag import file_io

# Guards API and CLI input from expanding into an unbounded Cartesian workload.
MAX_MATRIX_RUNS = 128


@dataclasses.dataclass(frozen=True, slots=True)
class MatrixAxis:
    """One dimension a comparison can sweep.

    Attributes:
      name: Key this axis uses in a request payload and in the report.
      field: `AppConfig` field a cell overrides with the axis value.
      flag: CLI flag that collects values for it.
      noun: Plural noun for the CLI help text.
      label_prefix: Prefix that distinguishes this axis inside a run label.
      choices: Allowed values, or None for an integer axis.
      default: Values used when a caller does not name any.
    """

    name: str
    field: str
    flag: str
    noun: str
    label_prefix: str
    choices: tuple[str, ...] | None
    default: Callable[[app_config.AppConfig], Sequence[object]]


def default_vector_backends(
    config: app_config.AppConfig,
) -> list[constants.VectorBackendName]:
    """Return the vector backends worth comparing for `config`.

    Args:
      config: The settings to read the Qdrant URL from.

    Returns:
      Qdrant is included only when a URL is configured; otherwise every
      Qdrant cell would report the same `missing_qdrant_url` skip.
    """
    return ["local", "qdrant"] if config.qdrant_url else ["local"]


AXES: tuple[MatrixAxis, ...] = (
    MatrixAxis(
        name="runtimes",
        field="agent_runtime",
        flag="--runtime",
        noun="runtimes",
        label_prefix="",
        choices=constants.AGENT_RUNTIMES,
        default=lambda config: [config.agent_runtime],
    ),
    MatrixAxis(
        name="vector_backends",
        field="vector_backend",
        flag="--vector-backend",
        noun="vector backends",
        label_prefix="",
        choices=constants.VECTOR_BACKENDS,
        default=default_vector_backends,
    ),
    MatrixAxis(
        name="chunk_strategies",
        field="chunk_strategy",
        flag="--chunk-strategy",
        noun="chunk strategies",
        label_prefix="",
        choices=constants.CHUNK_STRATEGIES,
        default=lambda config: list(constants.CHUNK_STRATEGIES),
    ),
    MatrixAxis(
        name="retrieval_strategies",
        field="retrieval_strategy",
        flag="--retrieval-strategy",
        noun="retrieval strategies",
        label_prefix="",
        choices=constants.RETRIEVAL_STRATEGIES,
        default=lambda config: list(constants.RETRIEVAL_STRATEGIES),
    ),
    MatrixAxis(
        name="rerankers",
        field="reranker",
        flag="--reranker",
        noun="rerankers",
        label_prefix="rr-",
        choices=constants.RERANKERS,
        # The one axis that does not sweep by default. Each `llm` cell spends a
        # model call per eval case, so sweeping unasked would turn a routine
        # comparison into an unrequested bill.
        default=lambda config: [config.reranker],
    ),
    MatrixAxis(
        name="top_ks",
        field="top_k",
        flag="--top-k",
        noun="top-k values",
        label_prefix="k",
        choices=None,
        default=lambda config: [config.top_k],
    ),
    MatrixAxis(
        name="chunk_sizes",
        field="chunk_size",
        flag="--chunk-size",
        noun="chunk sizes",
        label_prefix="s",
        choices=None,
        default=lambda config: [config.chunk_size],
    ),
    MatrixAxis(
        name="chunk_overlaps",
        field="chunk_overlap",
        flag="--chunk-overlap",
        noun="chunk overlap values",
        label_prefix="o",
        choices=None,
        default=lambda config: [config.chunk_overlap],
    ),
)


def resolve_axes(
    config: app_config.AppConfig,
    requested: Mapping[str, Sequence[object] | None],
) -> dict[str, list[object]]:
    """Return the values every axis will sweep.

    Args:
      config: The baseline settings, which supply each axis's default.
      requested: Values named by the caller, keyed by axis name. A key that
        is absent or None takes the axis default; an explicitly empty
        sequence is left empty, so it is rejected rather than silently
        widened back to the default.

    Returns:
      Every axis's values, keyed by axis name.
    """
    resolved: dict[str, list[object]] = {}
    for axis in AXES:
        values = requested.get(axis.name)
        resolved[axis.name] = list(
            axis.default(config) if values is None else values
        )
    return resolved


def run_eval_matrix(
    config: app_config.AppConfig,
    requested: Mapping[str, Sequence[object] | None] | None = None,
    output_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Evaluate every combination of the requested axes and rank the results.

    Args:
      config: The baseline settings each cell varies from.
      requested: Values to sweep, keyed by axis name. An omitted axis takes
        its default, which for most axes is the configured value held
        steady.
      output_path: Where to write the report, or None to skip writing it.

    Returns:
      The report: every cell that ran, and the leaderboard built from the
      ones that succeeded.

    Raises:
      ConfigurationError: If an axis is empty, or the Cartesian product
        would exceed `MAX_MATRIX_RUNS`, so one request cannot start an
        unbounded run.
    """
    axes = resolve_axes(config, requested or {})
    values = [axes[axis.name] for axis in AXES]
    if any(not entry for entry in values):
        raise exceptions.ConfigurationError(
            "Eval comparison axes must not be empty"
        )
    num_combinations = math.prod(len(entry) for entry in values)
    if num_combinations > MAX_MATRIX_RUNS:
        raise exceptions.ConfigurationError(
            f"Eval comparison requested {num_combinations} runs; "
            f"maximum is {MAX_MATRIX_RUNS}",
            action_hint="Reduce one or more comparison axes.",
        )

    runs: list[dict[str, object]] = []
    for combination in itertools.product(*values):
        # `strict=True` is the guard the parallel lists never had: an axis
        # added to `AXES` but missed here cannot bind to the wrong field.
        overrides = {
            axis.field: value
            for axis, value in zip(AXES, combination, strict=True)
        }
        runs.append(_run_matrix_case(config.with_overrides(**overrides)))

    payload: dict[str, object] = {
        "num_runs": len(runs),
        **axes,
        "leaderboard": _build_leaderboard(runs),
        "runs": runs,
    }
    if output_path is not None:
        file_io.atomic_write_text(output_path, _dump_json(payload))
    return payload


def _dump_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _run_matrix_case(config: app_config.AppConfig) -> dict[str, object]:
    label = run_label(config)
    common = {
        "label": label,
        "retrieval_config": presenters.serialize_retrieval_config(config),
        "runtime": config.agent_runtime,
    }
    skip_reason = _skip_reason(config)
    if skip_reason is not None:
        return {**common, "status": "skipped", "reason": skip_reason}

    try:
        rag.ingest_documents(config)
        results = harness.run_eval(config)
        summary = presenters.serialize_eval_summary(
            results,
            runtime=config.agent_runtime,
            config=config,
        )
        return {"label": label, "status": "ok", "summary": summary}
    except Exception as exc:
        skip_reason = _qdrant_runtime_skip_reason(config, exc)
        if skip_reason is not None:
            return {**common, "status": "skipped", "reason": skip_reason}
        return {
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_label(config: app_config.AppConfig) -> str:
    """Return the label identifying one cell of the matrix.

    Args:
      config: The settings that cell runs under.

    Returns:
      One segment per axis, so two cells that differ in any axis cannot
      share a label. Deriving it from `AXES` is what keeps that true: a new
      axis joins the label without being remembered separately.
    """
    return ":".join(
        f"{axis.label_prefix}{getattr(config, axis.field)}" for axis in AXES
    )


def _skip_reason(config: app_config.AppConfig) -> str | None:
    if config.vector_backend == "qdrant" and not config.qdrant_url:
        return "missing_qdrant_url"
    return None


def _qdrant_runtime_skip_reason(
    config: app_config.AppConfig, exc: Exception
) -> str | None:
    if config.vector_backend != "qdrant":
        return None

    if isinstance(exc, exceptions.VectorStoreError):
        if exc.reason_code == "unreachable":
            return "qdrant_unreachable"
        if exc.reason_code == "dependency_missing":
            return "missing_qdrant_client"
    return None


def _build_leaderboard(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    leaderboard: list[dict[str, object]] = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        summary = run.get("summary")
        if not isinstance(summary, dict):
            continue
        leaderboard.append(
            {
                "label": str(run["label"]),
                "answer_keyword_hit_rate": summary.get(
                    "answer_keyword_hit_rate", 0.0
                ),
                "retrieval_source_hit_rate": summary.get(
                    "retrieval_source_hit_rate", 0.0
                ),
                "retrieval_span_hit_rate": summary.get(
                    "retrieval_span_hit_rate", 0.0
                ),
                "citation_span_hit_rate": summary.get(
                    "citation_span_hit_rate", 0.0
                ),
                "avg_response_time_ms": summary.get(
                    "avg_response_time_ms", 0.0
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            _as_float(row["retrieval_span_hit_rate"]),
            _as_float(row["answer_keyword_hit_rate"]),
            _as_float(row["citation_span_hit_rate"]),
            -_as_float(row["avg_response_time_ms"]),
        ),
        reverse=True,
    )
    return leaderboard


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
