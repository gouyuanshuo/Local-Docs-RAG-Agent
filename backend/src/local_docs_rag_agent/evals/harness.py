from __future__ import annotations

import json
import time
from pathlib import Path

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import EvalCase, EvalResult


def load_eval_cases(eval_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cases.append(EvalCase(**payload))
    return cases


def run_eval(config: AppConfig) -> list[EvalResult]:
    agent = LocalDocsAgent(config)
    cases = load_eval_cases(config.eval_path)
    results: list[EvalResult] = []

    for case in cases:
        started_at = time.perf_counter()
        response = agent.answer(case.question)
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
        answer_lower = response.answer.lower()
        span_text = " ".join(span.text for span in response.citation_spans).lower()

        keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in answer_lower)
        keyword_hit_rate = keyword_hits / len(case.expected_keywords) if case.expected_keywords else 0.0
        source_hit = any(source in response.citations for source in case.expected_sources)
        span_hits = sum(
            1 for keyword in case.expected_span_keywords if keyword.lower() in span_text
        )
        citation_span_hit_rate = (
            span_hits / len(case.expected_span_keywords) if case.expected_span_keywords else 0.0
        )

        results.append(
            EvalResult(
                question=case.question,
                answer=response.answer,
                citations=response.citations,
                keyword_hit_rate=keyword_hit_rate,
                source_hit=source_hit,
                citation_span_hit_rate=citation_span_hit_rate,
                response_time_ms=response_time_ms,
            )
        )

    return results
