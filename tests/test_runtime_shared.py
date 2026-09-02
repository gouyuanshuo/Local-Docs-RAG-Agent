from __future__ import annotations

from local_docs_rag_agent.models import DocumentChunk, RetrievalHit
from local_docs_rag_agent.providers.chat import OpenAICompatibleChatProvider
from local_docs_rag_agent.rag.scoring import build_retrieval_hit
from local_docs_rag_agent.runtime.shared import (
    build_answer_context,
    collect_citations,
    format_tool_search_results,
    merge_hits,
)

MULTILINE_BODY = "# Attention\n\nAttention maps queries against keys and values."


def _hit(
    chunk_id: str = "doc.md::chunk-0",
    source_path: str = "doc.md",
    score: float = 0.9,
) -> RetrievalHit:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        source_path=source_path,
        title="Attention",
        text=MULTILINE_BODY,
        chunk_index=0,
        start_char=0,
        end_char=len(MULTILINE_BODY),
    )
    return build_retrieval_hit(chunk, score)


def test_context_labels_start_at_column_zero() -> None:
    context = build_answer_context([_hit()])

    labelled_lines = [
        line
        for line in context.splitlines()
        if line.startswith(("[S", "source:", "title:", "chunk_index:", "content:"))
    ]
    assert len(labelled_lines) == 5
    assert not any(line.startswith(" ") for line in context.splitlines() if line.startswith("["))
    assert "content: # Attention" in context


def test_extractive_fallback_recovers_context_from_a_multiline_chunk() -> None:
    # A chunk body almost always contains a line at column 0. If the context template
    # leaves its labels indented, the fallback parser finds nothing and wrongly claims
    # there was no supporting context even though retrieval succeeded.
    provider = OpenAICompatibleChatProvider(api_key=None, model="unused")

    answer = provider.answer(
        question="How is attention explained?",
        context=build_answer_context([_hit()]),
    )

    assert "could not find supporting context" not in answer
    assert "Attention maps queries against keys and values." in answer
    assert provider.status.mode == "fallback"


def test_fallback_still_reports_when_context_is_genuinely_empty() -> None:
    provider = OpenAICompatibleChatProvider(api_key=None, model="unused")

    answer = provider.answer(question="anything", context=build_answer_context([]))

    assert "could not find supporting context" in answer


def test_tool_results_include_the_score_and_full_text() -> None:
    rendered = format_tool_search_results([_hit(score=0.4242)])

    lines = rendered.splitlines()
    assert lines[0] == "[S1]"
    assert "score: 0.4242" in lines
    assert lines[-1] == "Attention maps queries against keys and values."


def test_merge_hits_keeps_first_occurrence_only() -> None:
    existing = [_hit(chunk_id="a")]

    merge_hits(existing, [_hit(chunk_id="a"), _hit(chunk_id="b", source_path="other.md")])

    assert [hit.chunk.chunk_id for hit in existing] == ["a", "b"]
    assert collect_citations(existing) == ["doc.md", "other.md"]
