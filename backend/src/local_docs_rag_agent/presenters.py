from __future__ import annotations

from local_docs_rag_agent.models import AgentAnswer, EvalResult


def serialize_answer(answer: AgentAnswer, runtime: str) -> dict[str, object]:
    return {
        "question": answer.question,
        "answer": answer.answer,
        "citations": answer.citations,
        "citation_spans": [
            {
                "source_path": span.source_path,
                "chunk_id": span.chunk_id,
                "chunk_index": span.chunk_index,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "text": span.text,
            }
            for span in answer.citation_spans
        ],
        "runtime": runtime,
        "diagnostics": {
            "requested_runtime": answer.diagnostics.requested_runtime,
            "actual_runtime": answer.diagnostics.actual_runtime,
            "vector_backend": answer.diagnostics.vector_backend,
            "chat_provider": {
                "provider": answer.diagnostics.chat_provider.provider,
                "mode": answer.diagnostics.chat_provider.mode,
                "reason": answer.diagnostics.chat_provider.reason,
            },
            "embedding_provider": {
                "provider": answer.diagnostics.embedding_provider.provider,
                "mode": answer.diagnostics.embedding_provider.mode,
                "reason": answer.diagnostics.embedding_provider.reason,
            },
        },
    }


def _avg(results: list[EvalResult], attr: str, precision: int = 4) -> float:
    if not results:
        return 0.0
    return round(sum(getattr(result, attr) for result in results) / len(results), precision)


def serialize_eval_summary(results: list[EvalResult], runtime: str) -> dict[str, object]:
    return {
        "num_cases": len(results),
        "runtime": runtime,
        "answer_keyword_hit_rate": _avg(results, "answer_keyword_hit_rate"),
        "retrieval_source_hit_rate": _avg(results, "retrieval_source_hit_rate"),
        "retrieval_span_hit_rate": _avg(results, "retrieval_span_hit_rate"),
        "citation_source_hit_rate": _avg(results, "citation_source_hit_rate"),
        "citation_span_hit_rate": _avg(results, "citation_span_hit_rate"),
        "avg_response_time_ms": _avg(results, "response_time_ms", precision=2),
        # Backward-compatible aliases for older UI/consumers.
        "avg_keyword_hit_rate": _avg(results, "answer_keyword_hit_rate"),
        "source_hit_rate": _avg(results, "citation_source_hit_rate"),
        "avg_citation_span_hit_rate": _avg(results, "citation_span_hit_rate"),
        "results": [
            {
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
                "expected_source_paths": result.expected_source_paths,
                "expected_answer_keywords": result.expected_answer_keywords,
                "expected_span_keywords": result.expected_span_keywords,
                "expected_retrieval_keywords": result.expected_retrieval_keywords,
                "failure_reasons": result.failure_reasons,
            }
            for result in results
        ],
    }
