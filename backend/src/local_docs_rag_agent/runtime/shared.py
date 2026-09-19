"""Prompt-context formatting and citation assembly shared by runtimes.

Keeping these here is what lets two runtimes produce answers with identical
citation semantics, so an eval can compare them on answer quality alone.
Retrieval itself lives in `rag.pipeline`, which both runtimes call directly: it
is a retrieval concern, not a runtime one, and putting it here would have made
the reranking stage look like something a runtime could choose to skip.

Both formatters render one labelled block per hit, `[S1]`, `[S2]`, and so on,
with the multi-line body last. That layout is a contract, not a style choice:
the extractive fallback in `providers/chat.py` parses these blocks back out when
no live model is available, and it can only do that if every label starts at
column 0 and the body is the final field in its block.
"""

from __future__ import annotations

import dataclasses

from local_docs_rag_agent.core import models


def build_answer_context(
    hits: list[models.RetrievalHit],
    source_ids: dict[str, str] | None = None,
) -> str:
    """Return hits rendered as the context handed to a chat provider.

    Args:
      hits: Retrieved evidence in the order shown to the provider.
      source_ids: Optional answer-local chunk-id mapping to populate and reuse.

    Returns:
      Labelled evidence blocks, or an explicit no-evidence message.
    """
    if not hits:
        return "No supporting documents were retrieved."
    active_source_ids = source_ids if source_ids is not None else {}
    return "\n\n".join(
        _render_hit_block(
            _source_id(hit, active_source_ids),
            fields=(
                ("source", hit.chunk.source_path),
                ("title", hit.chunk.title),
                ("chunk_index", str(hit.chunk.chunk_index)),
                ("start_char", str(hit.citation_span.start_char)),
                ("end_char", str(hit.citation_span.end_char)),
            ),
            body_label="content",
            body=hit.chunk.text,
        )
        for hit in hits
    )


def format_tool_search_results(
    hits: list[models.RetrievalHit],
    source_ids: dict[str, str] | None = None,
) -> str:
    """Return hits rendered as the agent search tool's result.

    Args:
      hits: Retrieved evidence in search-result order.
      source_ids: Optional run-local chunk-id mapping to populate and reuse.

    Returns:
      Labelled result blocks, or an explicit no-results message.
    """
    if not hits:
        return "No relevant chunks were found."
    active_source_ids = source_ids if source_ids is not None else {}
    return "\n\n".join(
        _render_hit_block(
            _source_id(hit, active_source_ids),
            fields=(
                ("source_path", hit.chunk.source_path),
                ("title", hit.chunk.title),
                ("chunk_index", str(hit.chunk.chunk_index)),
                ("start_char", str(hit.citation_span.start_char)),
                ("end_char", str(hit.citation_span.end_char)),
                ("score", f"{hit.score:.4f}"),
            ),
            body_label="text",
            body=hit.chunk.text,
        )
        for hit in hits
    )


def collect_citations(hits: list[models.RetrievalHit]) -> list[str]:
    """Return the cited source paths, de-duplicated and in retrieval order."""
    seen: set[str] = set()
    citations: list[str] = []
    for hit in hits:
        if hit.chunk.source_path in seen:
            continue
        seen.add(hit.chunk.source_path)
        citations.append(hit.chunk.source_path)
    return citations


def collect_citation_spans(
    hits: list[models.RetrievalHit],
    source_ids: dict[str, str] | None = None,
) -> list[models.CitationSpan]:
    """Return answer-owned source spans in retrieval order.

    Args:
      hits: Retrieved evidence whose spans back the answer.
      source_ids: Optional answer-local chunk-id mapping to populate and reuse.

    Returns:
      Copies of the retrieval spans carrying their answer-local source ids.
    """
    active_source_ids = source_ids if source_ids is not None else {}
    return [
        dataclasses.replace(
            hit.citation_span,
            source_id=_source_id(hit, active_source_ids),
        )
        for hit in hits
    ]


def merge_hits(
    existing: list[models.RetrievalHit], new_hits: list[models.RetrievalHit]
) -> None:
    """Append hits not already present, so tool searches accumulate."""
    seen = {hit.chunk.chunk_id for hit in existing}
    for hit in new_hits:
        if hit.chunk.chunk_id in seen:
            continue
        seen.add(hit.chunk.chunk_id)
        existing.append(hit)


def build_agent_answer(
    question: str,
    answer: str,
    hits: list[models.RetrievalHit],
    diagnostics: models.AnswerDiagnostics,
    source_ids: dict[str, str] | None = None,
) -> models.AgentAnswer:
    """Return the final answer, citing the hits that produced it.

    Args:
      question: The question that was answered.
      answer: The generated answer text.
      hits: Retrieved evidence accumulated for the answer.
      diagnostics: Runtime and provider behavior behind the answer.
      source_ids: Optional model-visible chunk-id mapping to preserve.

    Returns:
      The answer with copied citation spans carrying model-visible ids.
    """
    return models.AgentAnswer(
        question=question,
        answer=answer,
        citations=collect_citations(hits),
        citation_spans=collect_citation_spans(hits, source_ids),
        retrieved_chunks=hits,
        diagnostics=diagnostics,
    )


def _source_id(hit: models.RetrievalHit, source_ids: dict[str, str]) -> str:
    source_id = source_ids.get(hit.chunk.chunk_id)
    if source_id is None:
        source_id = f"S{len(source_ids) + 1}"
        source_ids[hit.chunk.chunk_id] = source_id
    return source_id


def _render_hit_block(
    source_id: str,
    *,
    fields: tuple[tuple[str, str], ...],
    body_label: str,
    body: str,
) -> str:
    # Built by explicit joining rather than an indented template: a chunk body
    # almost always contains a line starting at column 0, which stops
    # `textwrap.dedent` from removing the template's indentation and would leave
    # every label indented.
    lines = [f"[{source_id}]"]
    lines.extend(f"{label}: {value}" for label, value in fields)
    lines.append(f"{body_label}: {body}")
    return "\n".join(lines)
