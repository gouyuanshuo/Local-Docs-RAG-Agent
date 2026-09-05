"""Prompt-context formatting and citation assembly shared by runtimes.

Keeping these here is what lets two runtimes produce answers with identical citation
semantics, so an eval can compare them on answer quality alone. Retrieval itself lives
in `rag.pipeline`, which both runtimes call directly: it is a retrieval concern, not a
runtime one, and putting it here would have made the reranking stage look like
something a runtime could choose to skip.

Both formatters render one labelled block per hit, `[S1]`, `[S2]`, and so on, with the
multi-line body last. That layout is a contract, not a style choice: the extractive
fallback in `providers/chat.py` parses these blocks back out when no live model is
available, and it can only do that if every label starts at column 0 and the body is
the final field in its block.
"""

from __future__ import annotations

from local_docs_rag_agent.models import (
    AgentAnswer,
    AnswerDiagnostics,
    CitationSpan,
    RetrievalHit,
)


def build_answer_context(hits: list[RetrievalHit]) -> str:
    """Render retrieved hits as the grounding context handed to a chat provider."""

    if not hits:
        return "No supporting documents were retrieved."
    return "\n\n".join(
        _render_hit_block(
            index,
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
        for index, hit in enumerate(hits, start=1)
    )


def format_tool_search_results(hits: list[RetrievalHit]) -> str:
    """Render retrieved hits as the return value of the agent's search tool."""

    if not hits:
        return "No relevant chunks were found."
    return "\n\n".join(
        _render_hit_block(
            index,
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
        for index, hit in enumerate(hits, start=1)
    )


def collect_citations(hits: list[RetrievalHit]) -> list[str]:
    """Return the cited source paths, de-duplicated and in retrieval order."""

    seen: set[str] = set()
    citations: list[str] = []
    for hit in hits:
        if hit.chunk.source_path in seen:
            continue
        seen.add(hit.chunk.source_path)
        citations.append(hit.chunk.source_path)
    return citations


def collect_citation_spans(hits: list[RetrievalHit]) -> list[CitationSpan]:
    """Return the exact source spans backing each hit, in retrieval order."""

    return [hit.citation_span for hit in hits]


def merge_hits(existing: list[RetrievalHit], new_hits: list[RetrievalHit]) -> None:
    """Append hits not already present, so repeated tool searches accumulate evidence."""

    seen = {hit.chunk.chunk_id for hit in existing}
    for hit in new_hits:
        if hit.chunk.chunk_id in seen:
            continue
        seen.add(hit.chunk.chunk_id)
        existing.append(hit)


def build_agent_answer(
    question: str,
    answer: str,
    hits: list[RetrievalHit],
    diagnostics: AnswerDiagnostics,
) -> AgentAnswer:
    """Assemble the final answer with citations derived from the hits that produced it."""

    return AgentAnswer(
        question=question,
        answer=answer,
        citations=collect_citations(hits),
        citation_spans=collect_citation_spans(hits),
        retrieved_chunks=hits,
        diagnostics=diagnostics,
    )


def _render_hit_block(
    index: int,
    *,
    fields: tuple[tuple[str, str], ...],
    body_label: str,
    body: str,
) -> str:
    # Built by explicit joining rather than an indented template: a chunk body almost
    # always contains a line starting at column 0, which stops `textwrap.dedent` from
    # removing the template's indentation and would leave every label indented.
    lines = [f"[S{index}]"]
    lines.extend(f"{label}: {value}" for label, value in fields)
    lines.append(f"{body_label}: {body}")
    return "\n".join(lines)
