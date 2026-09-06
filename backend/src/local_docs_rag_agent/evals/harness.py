"""Loads eval cases and scores one run of the agent against them.

The harness measures five things separately, because they fail for different
reasons and a single blended score would hide which part broke:

* `retrieval_source_hit_rate` — did retrieval find the right documents at all?
* `retrieval_span_hit_rate` — did the retrieved text contain the expected
  evidence?
* `answer_keyword_hit_rate` — did the answer say what it should?
* `citation_source_hit_rate` — did the answer cite the right documents?
* `citation_span_hit_rate` — did the cited spans contain the expected evidence?

An expectation left empty scores 1.0 rather than 0.0, so a case can assert on
citation quality without being forced to also assert on answer wording.
"""

from __future__ import annotations

import json
import pathlib
import time

from local_docs_rag_agent import agent, exceptions, models
from local_docs_rag_agent import config as app_config


def load_eval_cases(eval_path: pathlib.Path) -> list[models.EvalCase]:
    """Read a JSONL eval file, reporting the line number of the first bad case."""

    cases: list[models.EvalCase] = []
    line_number: int | str = "unknown"
    try:
        with eval_path.open("r", encoding="utf-8") as handle:
            for current_line_number, line in enumerate(handle, start=1):
                line_number = current_line_number
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise exceptions.DataFormatError(
                        "Eval case must be a JSON object"
                    )
                cases.append(_normalize_eval_case(payload))
    except exceptions.DataFormatError as exc:
        raise exceptions.DataFormatError(
            f"Invalid eval file {eval_path} at line {line_number}: {exc}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise exceptions.DataFormatError(
            f"Could not read eval file {eval_path} at line {line_number}: {exc}"
        ) from exc
    return cases


def run_eval(config: app_config.AppConfig) -> list[models.EvalResult]:
    """Answer every eval case under `config` and score the results."""

    docs_agent = agent.LocalDocsAgent(config)
    results: list[models.EvalResult] = []

    for case in load_eval_cases(config.eval_path):
        started_at = time.perf_counter()
        response = docs_agent.answer(case.question)
        response_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

        answer_text = response.answer.lower()
        citation_text = " ".join(
            span.text for span in response.citation_spans
        ).lower()
        retrieved_text = " ".join(
            hit.chunk.text for hit in response.retrieved_chunks
        ).lower()
        retrieved_sources = list(
            dict.fromkeys(
                hit.chunk.source_path for hit in response.retrieved_chunks
            )
        )

        result = models.EvalResult(
            question=case.question,
            answer=response.answer,
            citations=response.citations,
            retrieved_sources=retrieved_sources,
            answer_keyword_hit_rate=keyword_match_rate(
                case.expected_answer_keywords, answer_text
            ),
            retrieval_source_hit_rate=source_match_rate(
                case.expected_source_paths, retrieved_sources
            ),
            retrieval_span_hit_rate=keyword_match_rate(
                case.expected_retrieval_keywords, retrieved_text
            ),
            citation_source_hit_rate=source_match_rate(
                case.expected_source_paths, response.citations
            ),
            citation_span_hit_rate=keyword_match_rate(
                case.expected_span_keywords, citation_text
            ),
            response_time_ms=response_time_ms,
            diagnostics=response.diagnostics,
            expected_source_paths=case.expected_source_paths,
            expected_answer_keywords=case.expected_answer_keywords,
            expected_span_keywords=case.expected_span_keywords,
            expected_retrieval_keywords=case.expected_retrieval_keywords,
        )
        result.failure_reasons = failure_reasons(result)
        results.append(result)

    return results


def keyword_match_rate(expected_items: list[str], observed_text: str) -> float:
    """Return the fraction of expected keywords present in `observed_text`.

    An empty expectation is neutral success, so a case may assert on some
    dimensions without being penalised for the ones it leaves unspecified.
    """

    if not expected_items:
        return 1.0
    hits = sum(1 for item in expected_items if item.lower() in observed_text)
    return hits / len(expected_items)


def source_match_rate(
    expected_sources: list[str], observed_sources: list[str]
) -> float:
    """Return the fraction of expected source paths present in `observed_sources`."""

    if not expected_sources:
        return 1.0
    observed = set(observed_sources)
    hits = sum(1 for source in expected_sources if source in observed)
    return hits / len(expected_sources)


def failure_reasons(result: models.EvalResult) -> list[str]:
    """Name every dimension the case fell short on, for triage without re-running
    it.
    """

    reasons: list[str] = []
    if result.retrieval_source_hit_rate < 1.0:
        reasons.append("retrieval_missed_expected_source")
    if (
        result.retrieval_span_hit_rate < 1.0
        and result.expected_retrieval_keywords
    ):
        reasons.append("retrieval_missed_expected_span")
    if result.answer_keyword_hit_rate < 1.0 and result.expected_answer_keywords:
        reasons.append("answer_missing_expected_keywords")
    if result.citation_source_hit_rate < 1.0:
        reasons.append("citation_missed_expected_source")
    if result.citation_span_hit_rate < 1.0 and result.expected_span_keywords:
        reasons.append("citation_missed_expected_span")
    return reasons


def _normalize_eval_case(payload: dict[str, object]) -> models.EvalCase:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise exceptions.DataFormatError(
            "Eval case question must be a non-empty string"
        )
    # The shorter key names are the original field names, still accepted for old
    # files.
    answer_keywords = _string_list(
        payload.get(
            "expected_answer_keywords", payload.get("expected_keywords", [])
        ),
        "expected_answer_keywords",
    )
    source_paths = _string_list(
        payload.get(
            "expected_source_paths", payload.get("expected_sources", [])
        ),
        "expected_source_paths",
    )
    span_keywords = _string_list(
        payload.get("expected_span_keywords", []),
        "expected_span_keywords",
    )
    # Retrieval expectations default to the citation expectations, because
    # evidence that must appear in a citation must first have been retrieved.
    retrieval_keywords = _string_list(
        payload.get("expected_retrieval_keywords", span_keywords),
        "expected_retrieval_keywords",
    )
    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise exceptions.DataFormatError(
            "Eval case notes must be a string or null"
        )

    return models.EvalCase(
        question=question.strip(),
        expected_answer_keywords=answer_keywords,
        expected_source_paths=source_paths,
        expected_span_keywords=span_keywords,
        expected_retrieval_keywords=retrieval_keywords,
        notes=notes,
    )


def _string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise exceptions.DataFormatError(
            f"Eval case {field_name} must be a list of strings"
        )
    return [item for item in value if item]
