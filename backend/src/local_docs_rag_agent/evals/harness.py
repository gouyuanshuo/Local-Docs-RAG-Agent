from __future__ import annotations

import json
import time
from pathlib import Path

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import EvalCase, EvalResult


def _normalize_eval_case(payload: dict[str, object]) -> EvalCase:
    answer_keywords = payload.get("expected_answer_keywords") or payload.get("expected_keywords") or []
    source_paths = payload.get("expected_source_paths") or payload.get("expected_sources") or []
    span_keywords = payload.get("expected_span_keywords") or []
    retrieval_keywords = payload.get("expected_retrieval_keywords") or span_keywords

    return EvalCase(
        question=str(payload["question"]),
        expected_answer_keywords=[str(keyword) for keyword in answer_keywords],
        expected_source_paths=[str(source) for source in source_paths],
        expected_span_keywords=[str(keyword) for keyword in span_keywords],
        expected_retrieval_keywords=[str(keyword) for keyword in retrieval_keywords],
        notes=str(payload["notes"]) if payload.get("notes") is not None else None,
    )


def load_eval_cases(eval_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cases.append(_normalize_eval_case(payload))
    return cases


def _match_rate(expected_items: list[str], observed_text: str) -> float:
    if not expected_items:
        return 0.0
    hits = sum(1 for item in expected_items if item.lower() in observed_text)
    return hits / len(expected_items)


def _source_rate(expected_sources: list[str], observed_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    observed = set(observed_sources)
    hits = sum(1 for source in expected_sources if source in observed)
    return hits / len(expected_sources)


def _failure_reasons(result: EvalResult) -> list[str]:
    reasons: list[str] = []
    if result.retrieval_source_hit_rate < 1.0:
        reasons.append("retrieval_missed_expected_source")
    if result.retrieval_span_hit_rate < 1.0 and result.expected_retrieval_keywords:
        reasons.append("retrieval_missed_expected_span")
    if result.answer_keyword_hit_rate < 1.0 and result.expected_answer_keywords:
        reasons.append("answer_missing_expected_keywords")
    if result.citation_source_hit_rate < 1.0:
        reasons.append("citation_missed_expected_source")
    if result.citation_span_hit_rate < 1.0 and result.expected_span_keywords:
        reasons.append("citation_missed_expected_span")
    return reasons


def run_eval(config: AppConfig) -> list[EvalResult]:
    agent = LocalDocsAgent(config)
    cases = load_eval_cases(config.eval_path)
    results: list[EvalResult] = []

    for case in cases:
        started_at = time.perf_counter()
        response = agent.answer(case.question)
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

        answer_lower = response.answer.lower()
        citation_text = " ".join(span.text for span in response.citation_spans).lower()
        retrieved_text = " ".join(hit.chunk.text for hit in response.retrieved_chunks).lower()
        retrieved_sources = list(dict.fromkeys(hit.chunk.source_path for hit in response.retrieved_chunks))

        result = EvalResult(
            question=case.question,
            answer=response.answer,
            citations=response.citations,
            retrieved_sources=retrieved_sources,
            answer_keyword_hit_rate=_match_rate(case.expected_answer_keywords, answer_lower),
            retrieval_source_hit_rate=_source_rate(case.expected_source_paths, retrieved_sources),
            retrieval_span_hit_rate=_match_rate(case.expected_retrieval_keywords, retrieved_text),
            citation_source_hit_rate=_source_rate(case.expected_source_paths, response.citations),
            citation_span_hit_rate=_match_rate(case.expected_span_keywords, citation_text),
            response_time_ms=response_time_ms,
            expected_source_paths=case.expected_source_paths,
            expected_answer_keywords=case.expected_answer_keywords,
            expected_span_keywords=case.expected_span_keywords,
            expected_retrieval_keywords=case.expected_retrieval_keywords,
        )
        result.failure_reasons = _failure_reasons(result)
        results.append(result)

    return results
