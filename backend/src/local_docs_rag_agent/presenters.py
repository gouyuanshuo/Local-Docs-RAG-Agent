"""Converts internal dataclasses into JSON-ready payloads.

This is the one place that knows the shape of an outgoing payload. The API validates
those payloads against `api/schemas.py`, the CLI prints them, and the eval comparison
report embeds them, so keeping the conversion here is what stops three delivery paths
from describing the same answer three slightly different ways.

The serializers are deliberately small and composable: `serialize_answer` and
`serialize_eval_result` both reuse `serialize_diagnostics`, which is why an added
diagnostic field appears everywhere at once.
"""

from __future__ import annotations

from collections.abc import Iterable

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import (
    AgentAnswer,
    AnswerDiagnostics,
    CitationSpan,
    EvalResult,
    ProviderStatus,
)

RATE_PRECISION = 4
MILLISECOND_PRECISION = 2


def serialize_provider_status(status: ProviderStatus) -> dict[str, object]:
    """Serialize one provider's health, including why it degraded."""

    return {
        "provider": status.provider,
        "mode": status.mode,
        "reason": status.reason,
    }


def serialize_diagnostics(diagnostics: AnswerDiagnostics) -> dict[str, object]:
    """Serialize the requested-versus-actual runtime and both provider statuses."""

    return {
        "requested_runtime": diagnostics.requested_runtime,
        "actual_runtime": diagnostics.actual_runtime,
        "vector_backend": diagnostics.vector_backend,
        "chat_provider": serialize_provider_status(diagnostics.chat_provider),
        "embedding_provider": serialize_provider_status(diagnostics.embedding_provider),
        "reranker": serialize_provider_status(diagnostics.reranker),
    }


def serialize_citation_span(span: CitationSpan) -> dict[str, object]:
    """Serialize a cited span together with the exact source offsets it came from."""

    return {
        "source_path": span.source_path,
        "chunk_id": span.chunk_id,
        "chunk_index": span.chunk_index,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "text": span.text,
    }


def serialize_retrieval_config(config: AppConfig) -> dict[str, object]:
    """Serialize the retrieval settings a result was produced under.

    Attaching this to eval output is what makes two runs comparable after the fact,
    instead of leaving the reader to guess which settings produced which numbers.
    """

    return {
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
        "docs_dir": str(config.docs_dir),
        "docs_exclude_patterns": list(config.docs_exclude_patterns),
    }


def serialize_answer(answer: AgentAnswer) -> dict[str, object]:
    """Serialize one answered question with its citations and diagnostics."""

    return {
        "question": answer.question,
        "answer": answer.answer,
        "citations": answer.citations,
        "citation_spans": [serialize_citation_span(span) for span in answer.citation_spans],
        "runtime": answer.diagnostics.actual_runtime,
        "diagnostics": serialize_diagnostics(answer.diagnostics),
    }


def serialize_eval_result(result: EvalResult) -> dict[str, object]:
    """Serialize one eval case: its metrics, its expectations, and why it failed."""

    return {
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_sources": result.retrieved_sources,
        "answer_keyword_hit_rate": result.answer_keyword_hit_rate,
        "retrieval_source_hit_rate": result.retrieval_source_hit_rate,
        "retrieval_span_hit_rate": result.retrieval_span_hit_rate,
        "citation_source_hit_rate": result.citation_source_hit_rate,
        "citation_span_hit_rate": result.citation_span_hit_rate,
        "response_time_ms": result.response_time_ms,
        "diagnostics": (
            serialize_diagnostics(result.diagnostics) if result.diagnostics else None
        ),
        "expected_source_paths": result.expected_source_paths,
        "expected_answer_keywords": result.expected_answer_keywords,
        "expected_span_keywords": result.expected_span_keywords,
        "expected_retrieval_keywords": result.expected_retrieval_keywords,
        "failure_reasons": result.failure_reasons,
    }


def serialize_eval_summary(
    results: list[EvalResult],
    runtime: str,
    config: AppConfig | None = None,
) -> dict[str, object]:
    """Aggregate per-case eval results into one reportable summary."""

    answer_keyword_hit_rate = _average(result.answer_keyword_hit_rate for result in results)
    citation_source_hit_rate = _average(result.citation_source_hit_rate for result in results)
    citation_span_hit_rate = _average(result.citation_span_hit_rate for result in results)
    return {
        "num_cases": len(results),
        "runtime": runtime,
        "retrieval_config": serialize_retrieval_config(config) if config else None,
        "answer_keyword_hit_rate": answer_keyword_hit_rate,
        "retrieval_source_hit_rate": _average(
            result.retrieval_source_hit_rate for result in results
        ),
        "retrieval_span_hit_rate": _average(result.retrieval_span_hit_rate for result in results),
        "citation_source_hit_rate": citation_source_hit_rate,
        "citation_span_hit_rate": citation_span_hit_rate,
        "avg_response_time_ms": _average(
            (result.response_time_ms for result in results),
            precision=MILLISECOND_PRECISION,
        ),
        # Backward-compatible aliases for older UI and report consumers.
        "avg_keyword_hit_rate": answer_keyword_hit_rate,
        "source_hit_rate": citation_source_hit_rate,
        "avg_citation_span_hit_rate": citation_span_hit_rate,
        "results": [serialize_eval_result(result) for result in results],
    }


def _average(values: Iterable[float], precision: int = RATE_PRECISION) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), precision)
