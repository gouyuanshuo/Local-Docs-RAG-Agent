"""Runs the eval harness across a matrix of retrieval configurations.

Each cell re-ingests and re-evaluates under one `AppConfig` variant so the
results are comparable. A cell that cannot run is reported as `skipped` with a
reason, a cell that fails is reported as `error`, and a cell whose chat,
embedding, reranker, or runtime degraded is `degraded`. None of those is
silently treated as success: a leaderboard that ranks fallback answers is
worse than no leaderboard.

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
import hashlib
import itertools
import json
import math
import pathlib
import tempfile
import typing
import uuid
from collections.abc import Callable, Mapping, Sequence

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import presenters, rag
from local_docs_rag_agent.core import constants, exceptions, file_io, models
from local_docs_rag_agent.evals import harness

# Guards API and CLI input from expanding into an unbounded Cartesian workload.
MAX_MATRIX_RUNS = 128


class _IdentityDigest(typing.Protocol):
    """Minimal hash protocol used for canonical comparison identities."""

    def update(self, data: bytes, /) -> object:
        """Add bytes to the digest."""


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


@dataclasses.dataclass(frozen=True, slots=True)
class EvalMatrixPlan:
    """The fully validated configurations a comparison will execute.

    Attributes:
      axes: Resolved values for every matrix axis, keyed by request field.
      variants: One validated configuration per Cartesian-product cell.
      num_runs: The number of variants and therefore comparison cells.
    """

    axes: dict[str, list[object]]
    variants: list[app_config.AppConfig]
    num_runs: int


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
        # Sweeping all four is for comparing rankers on one backend. Qdrant
        # `blended` is dense-only; a local-vs-qdrant backend compare should
        # pass `dense` and `hybrid_rrf` explicitly, not `blended`.
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


def plan_eval_matrix(
    config: app_config.AppConfig,
    requested: Mapping[str, Sequence[object] | None] | None = None,
) -> EvalMatrixPlan:
    """Resolve and validate all configurations in an eval comparison.

    Args:
      config: The baseline settings each cell varies from.
      requested: Values to sweep, keyed by axis name. An omitted axis takes
        its default, which for most axes is the configured value held steady.

    Returns:
      The resolved axes, every validated `AppConfig` variant, and their count.

    Raises:
      ConfigurationError: If an axis is empty, the Cartesian product exceeds
        `MAX_MATRIX_RUNS`, or any resulting `AppConfig` is invalid.
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
    variants = [
        config.with_overrides(
            **{
                axis.field: value
                for axis, value in zip(AXES, combination, strict=True)
            }
        )
        for combination in itertools.product(*values)
    ]
    return EvalMatrixPlan(
        axes=axes,
        variants=variants,
        num_runs=len(variants),
    )


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
      ConfigurationError: If an axis is empty, the Cartesian product would
        exceed `MAX_MATRIX_RUNS`, or any planned configuration is invalid.
      DataFormatError: If the eval file contains no cases. The file is
        loaded before matrix ingestion can mutate an index.
    """
    plan = plan_eval_matrix(config, requested)
    harness.load_eval_cases(config.eval_path)
    dataset_identity = _dataset_identity(config)

    runs = [
        _run_matrix_case(variant, dataset_identity=dataset_identity)
        for variant in plan.variants
    ]

    payload: dict[str, object] = {
        "num_runs": plan.num_runs,
        "dataset_identity": dataset_identity,
        **plan.axes,
        "leaderboard": _build_leaderboard(runs),
        "runs": runs,
    }
    if output_path is not None:
        file_io.atomic_write_text(output_path, _dump_json(payload))
    return payload


def _dump_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _run_matrix_case(
    config: app_config.AppConfig,
    *,
    dataset_identity: str | None = None,
) -> dict[str, object]:
    label = run_label(config)
    common: dict[str, object] = {
        "label": label,
        "retrieval_config": presenters.serialize_retrieval_config(config),
        "runtime": config.agent_runtime,
        "configuration_identity": _configuration_identity(config),
    }
    if dataset_identity is not None:
        common["dataset_identity"] = dataset_identity
    skip_reason = _skip_reason(config)
    if skip_reason is not None:
        return {**common, "status": "skipped", "reason": skip_reason}

    if config.vector_backend == "qdrant":
        return _run_qdrant_matrix_case(config, common)
    return _run_local_matrix_case(config, common)


def _run_local_matrix_case(
    config: app_config.AppConfig,
    common: dict[str, object],
) -> dict[str, object]:
    """Evaluate one local cell in temporary storage.

    Returns:
      The completed, degraded, skipped, or error cell payload.
    """
    try:
        with tempfile.TemporaryDirectory(
            prefix="local-docs-rag-compare-"
        ) as storage_dir:
            storage_path = pathlib.Path(storage_dir)
            isolated = config.with_overrides(
                index_path=storage_path / "chunks.jsonl",
                ingest_manifest_path=storage_path / "ingest_manifest.json",
            )
            rag.ingest_documents(isolated)
            results = harness.run_eval(isolated)
            return _result_from_eval_results(results, config, common)
    except Exception as exc:
        return _error_result(common, config, exc)


def _run_qdrant_matrix_case(
    config: app_config.AppConfig,
    common: dict[str, object],
) -> dict[str, object]:
    """Evaluate and clean up one disposable Qdrant comparison cell.

    Returns:
      The completed, degraded, skipped, or error cell payload.
    """
    collection_name = _disposable_qdrant_collection(config.qdrant_collection)
    metadata: dict[str, object] = {
        "disposable_qdrant_collection": collection_name,
    }
    common = {**common, "run_metadata": metadata}
    ownership: rag.QdrantIndexOwnership | None = None
    cell_error: Exception | None = None
    result: dict[str, object]

    with tempfile.TemporaryDirectory(
        prefix="local-docs-rag-compare-"
    ) as storage_dir:
        isolated = config.with_overrides(
            qdrant_collection=collection_name,
            ingest_manifest_path=pathlib.Path(storage_dir)
            / "ingest_manifest.json",
        )
        try:
            ownership = rag.initialize_owned_qdrant_index(isolated)
            results = harness.run_eval(isolated)
            result = _result_from_eval_results(results, config, common)
        except Exception as exc:
            cell_error = exc
            result = _error_result(common, config, exc)

        if ownership is None:
            if _has_partial_cleanup_failure(cell_error):
                metadata["orphan_recovery_required"] = True
                return _initializer_cleanup_error_result(
                    common,
                    collection_name,
                    cell_error,
                )
            return result
        try:
            rag.delete_owned_qdrant_index(isolated, ownership)
        except Exception as cleanup_error:
            metadata["orphan_recovery_required"] = True
            return _cleanup_error_result(
                common,
                collection_name,
                cell_error,
                cleanup_error,
            )
    return result


def _result_from_eval_results(
    results: list[models.EvalResult],
    original_config: app_config.AppConfig,
    common: dict[str, object],
) -> dict[str, object]:
    """Return one cell payload using the original semantic configuration."""
    summary = presenters.serialize_eval_summary(
        results,
        runtime=original_config.agent_runtime,
        config=original_config,
    )
    degraded = cell_degradation_reason(results)
    if degraded is not None:
        return {
            **common,
            "status": "degraded",
            "reason": degraded,
            "summary": summary,
        }
    return {**common, "status": "ok", "summary": summary}


def _error_result(
    common: dict[str, object],
    config: app_config.AppConfig,
    exc: Exception,
) -> dict[str, object]:
    """Return the existing skip or error representation for one cell failure."""
    skip_reason = _qdrant_runtime_skip_reason(config, exc)
    if skip_reason is not None:
        return {**common, "status": "skipped", "reason": skip_reason}
    return {
        **common,
        "status": "error",
        "error": _exception_description(exc),
    }


def _cleanup_error_result(
    common: dict[str, object],
    collection_name: str,
    cell_error: Exception | None,
    cleanup_error: Exception,
) -> dict[str, object]:
    """Return a visible cleanup error without discarding the cell failure."""
    message = (
        "Qdrant comparison cleanup failed for disposable collection "
        f"{collection_name}: {type(cleanup_error).__name__}: {cleanup_error}"
    )
    if cell_error is not None:
        message += (
            f"; original cell failure: {_exception_description(cell_error)}"
        )
    return {**common, "status": "error", "error": message}


def _initializer_cleanup_error_result(
    common: dict[str, object],
    collection_name: str,
    cell_error: Exception | None,
) -> dict[str, object]:
    """Report a failed initializer cleanup with exact orphan recovery data.

    Returns:
      An error payload with the exact disposable collection name.
    """
    if cell_error is None:
        raise AssertionError("Initializer cleanup report requires a cell error")
    return {
        **common,
        "status": "error",
        "error": (
            "Qdrant comparison initializer reported failed partial cleanup "
            f"for disposable collection {collection_name}: "
            f"{_exception_description(cell_error)}"
        ),
    }


def _has_partial_cleanup_failure(exc: Exception | None) -> bool:
    """Return whether an initializer reports a failed partial cleanup note."""
    if exc is None:
        return False
    return any(
        "partial" in note.lower()
        and "cleanup" in note.lower()
        and "fail" in note.lower()
        for note in getattr(exc, "__notes__", ())
    )


def _exception_description(exc: Exception) -> str:
    """Return an exception message including non-secret diagnostic notes."""
    message = f"{type(exc).__name__}: {exc}"
    notes = getattr(exc, "__notes__", ())
    if notes:
        message += "; notes: " + " | ".join(notes)
    return message


def _disposable_qdrant_collection(baseline: str) -> str:
    """Return a bounded UUID collection name distinct from `baseline`."""
    safe_baseline = "".join(
        character if character.isalnum() or character in "_-" else "-"
        for character in baseline
    ).strip("-_")
    prefix = safe_baseline[:48] or "local-docs-rag"
    return f"{prefix}-eval-{uuid.uuid4().hex}"


def _dataset_identity(config: app_config.AppConfig) -> str:
    """Hash corpus and gold checksums without returning their contents.

    Returns:
      A stable non-secret digest of the evaluation dataset.
    """
    digest = hashlib.sha256(b"local-docs-rag-dataset-v1\0")
    source_texts = rag.read_source_texts(
        config.docs_dir, config.docs_exclude_patterns
    )
    for source_path, text in sorted(source_texts.items()):
        _update_identity_digest(
            digest,
            _canonical_source_path(config.docs_dir, source_path),
        )
        _update_identity_digest(
            digest, hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
    _update_identity_digest(
        digest, hashlib.sha256(config.eval_path.read_bytes()).hexdigest()
    )
    return digest.hexdigest()


def _canonical_source_path(docs_dir: pathlib.Path, source_path: str) -> str:
    """Return a corpus-root-relative source identity when possible."""
    try:
        return (
            pathlib.Path(source_path)
            .resolve()
            .relative_to(docs_dir.resolve())
            .as_posix()
        )
    except ValueError:
        return pathlib.Path(source_path).as_posix()


def _configuration_identity(config: app_config.AppConfig) -> str:
    """Hash semantic comparison settings without keys, URLs, or paths.

    Returns:
      A stable non-secret digest of planned semantic settings.
    """
    values = {
        "agent_runtime": config.agent_runtime,
        "agents_max_turns": config.agents_max_turns,
        "vector_backend": config.vector_backend,
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "retrieval_strategy": config.retrieval_strategy,
        "retrieval_candidate_k": config.retrieval_candidate_k,
        "rrf_k": config.rrf_k,
        "reranker": config.reranker,
        "rerank_candidate_k": config.rerank_candidate_k,
        "rerank_model": config.rerank_model,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "llm_api_style": config.llm_api_style,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "embedding_batch_size": config.embedding_batch_size,
        "embedding_max_retries": config.embedding_max_retries,
        "embedding_retry_backoff_ms": config.embedding_retry_backoff_ms,
    }
    encoded = json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(b"local-docs-rag-config-v1\0" + encoded).hexdigest()


def _update_identity_digest(
    digest: _IdentityDigest,
    value: str,
) -> None:
    """Add one length-delimited canonical string to an identity digest."""
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def cell_degradation_reason(
    results: list[models.EvalResult],
) -> str | None:
    """Return why a finished cell must not enter the leaderboard.

    Chat and embedding must provide live execution evidence, and a selected
    reranker must do the same. A disabled reranker alone may report `ready`
    with `reranker_disabled`; missing or unknown diagnostics are not evidence
    of a live run. Ranking fallback or unmeasured cells as `ok` would misstate
    the result.

    Args:
      results: Per-case eval outcomes, including diagnostics.

    Returns:
      A stable comma-separated reason, or None when every case supplied live
      diagnostics for the requested runtime. A disabled reranker may instead
      report `ready` with `reranker_disabled`.
    """
    if not results:
        return "no_eval_cases"
    reasons: list[str] = []
    for result in results:
        diagnostics = result.diagnostics
        if diagnostics is None:
            return "missing_diagnostics"
        if diagnostics.requested_runtime != diagnostics.actual_runtime:
            reasons.append("runtime_fallback")
        if diagnostics.chat_provider.mode == "fallback":
            reasons.append("chat_fallback")
        elif diagnostics.chat_provider.mode != "live":
            reasons.append("chat_not_live")
        if diagnostics.embedding_provider.mode == "fallback":
            reasons.append("embedding_fallback")
        elif diagnostics.embedding_provider.mode != "live":
            reasons.append("embedding_not_live")
        if diagnostics.reranker.mode == "fallback":
            reasons.append("reranker_fallback")
        elif not _reranker_is_live_or_disabled(diagnostics.reranker):
            reasons.append("reranker_not_live")
    if not reasons:
        return None
    return ",".join(dict.fromkeys(reasons))


def _reranker_is_live_or_disabled(status: models.ProviderStatus) -> bool:
    """Return whether reranking was live or intentionally not selected."""
    return status.mode == "live" or (
        status.mode == "ready" and status.reason == "reranker_disabled"
    )


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
                "retrieval_reciprocal_rank": summary.get(
                    "retrieval_reciprocal_rank", 0.0
                ),
                "retrieval_precision": summary.get("retrieval_precision", 0.0),
                "citation_span_hit_rate": summary.get(
                    "citation_span_hit_rate", 0.0
                ),
                "avg_response_time_ms": summary.get(
                    "avg_response_time_ms", 0.0
                ),
            }
        )
    # Ranked by reciprocal rank first, not by the span hit rate. Span matching
    # cannot tell a configuration that puts evidence first from one that
    # buries it fourth, and a wider `top_k` can only raise it. Ranking on it
    # made the leaderboard prefer exactly the loose retrieval a second stage
    # exists to tighten.
    leaderboard.sort(
        key=lambda row: (
            _as_float(row["retrieval_reciprocal_rank"]),
            _as_float(row["retrieval_precision"]),
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
