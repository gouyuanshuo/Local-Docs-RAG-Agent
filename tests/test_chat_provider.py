from local_docs_rag_agent.providers import chat


def test_fallback_answer_keeps_multiline_chunk_content() -> None:
    provider = chat.OpenAICompatibleChatProvider(
        api_key=None,
        model="unused",
        provider_label="test",
    )
    context = """[S1]
source: first.md
title: First
content: first line
second line

[S2]
source: second.md
title: Second
content: another source
"""

    answer = provider.answer("question", context)

    assert "first line\nsecond line" in answer
    assert "another source" in answer
