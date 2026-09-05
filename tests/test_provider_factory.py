from __future__ import annotations

from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.providers.chat import OpenAICompatibleChatProvider
from local_docs_rag_agent.providers.embedding import OpenAICompatibleEmbeddingProvider
from local_docs_rag_agent.providers.factory import build_chat_provider, build_embedding_provider


def test_factory_threads_external_http_proxy_setting() -> None:
    config = AppConfig.from_env().with_overrides(
        llm_api_key=None,
        embedding_api_key=None,
        external_http_trust_env=False,
    )

    chat_provider = build_chat_provider(config)
    embedding_provider = build_embedding_provider(config)

    # Narrowing to the concrete types also asserts which providers the factory chose.
    assert isinstance(chat_provider, OpenAICompatibleChatProvider)
    assert isinstance(embedding_provider, OpenAICompatibleEmbeddingProvider)
    assert chat_provider._trust_env is False
    assert embedding_provider._trust_env is False
