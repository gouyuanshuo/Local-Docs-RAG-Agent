from __future__ import annotations

from textwrap import dedent

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.models import (
    AgentAnswer,
    AnswerDiagnostics,
    CitationSpan,
    ProviderStatus,
    RetrievalHit,
)
from local_docs_rag_agent.rag.ingest import build_store


def retrieve_hits(config: AppConfig, question: str) -> tuple[list[RetrievalHit], ProviderStatus]:
    store = build_store(config)
    hits = store.search(query=question, top_k=config.top_k)
    return hits, store.embedding_status


def build_answer_context(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "No supporting documents were retrieved."

    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        parts.append(
            dedent(
                f"""
                [S{index}]
                source: {hit.chunk.source_path}
                title: {hit.chunk.title}
                chunk_index: {hit.chunk.chunk_index}
                start_char: {hit.citation_span.start_char}
                end_char: {hit.citation_span.end_char}
                content: {hit.chunk.text}
                """
            ).strip()
        )
    return "\n\n".join(parts)


def format_tool_search_results(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "No relevant chunks were found."

    entries: list[str] = []
    for index, hit in enumerate(hits, start=1):
        entries.append(
            dedent(
                f"""
                [S{index}]
                source_path: {hit.chunk.source_path}
                title: {hit.chunk.title}
                chunk_index: {hit.chunk.chunk_index}
                start_char: {hit.citation_span.start_char}
                end_char: {hit.citation_span.end_char}
                score: {hit.score:.4f}
                text: {hit.chunk.text}
                """
            ).strip()
        )
    return "\n\n".join(entries)


def collect_citations(hits: list[RetrievalHit]) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for hit in hits:
        if hit.chunk.source_path in seen:
            continue
        seen.add(hit.chunk.source_path)
        citations.append(hit.chunk.source_path)
    return citations


def collect_citation_spans(hits: list[RetrievalHit]) -> list[CitationSpan]:
    return [hit.citation_span for hit in hits]


def merge_hits(existing: list[RetrievalHit], new_hits: list[RetrievalHit]) -> None:
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
    return AgentAnswer(
        question=question,
        answer=answer,
        citations=collect_citations(hits),
        citation_spans=collect_citation_spans(hits),
        retrieved_chunks=hits,
        diagnostics=diagnostics,
    )
