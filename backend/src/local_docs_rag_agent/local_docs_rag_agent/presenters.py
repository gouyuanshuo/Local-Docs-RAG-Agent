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
    }


def serialize_eval_summary(results: list[EvalResult], runtime: str) -> dict[str, object]:
    return {
        "num_cases": len(results),
        "runtime": runtime,
        "avg_keyword_hit_rate": round(
            sum(result.keyword_hit_rate for result in results) / len(results), 4
        )
        if results
        else 0.0,
        "source_hit_rate": round(
            sum(1 for result in results if result.source_hit) / len(results), 4
        )
        if results
        else 0.0,
        "avg_citation_span_hit_rate": round(
            sum(result.citation_span_hit_rate for result in results) / len(results), 4
        )
        if results
        else 0.0,
        "avg_response_time_ms": round(
            sum(result.response_time_ms for result in results) / len(results), 2
        )
        if results
        else 0.0,
        "results": [
            {
                "question": result.question,
                "keyword_hit_rate": result.keyword_hit_rate,
                "source_hit": result.source_hit,
                "citation_span_hit_rate": result.citation_span_hit_rate,
                "response_time_ms": result.response_time_ms,
                "citations": result.citations,
            }
            for result in results
        ],
    }
