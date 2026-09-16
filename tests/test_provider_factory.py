from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.providers import chat, embedding
from local_docs_rag_agent.providers import factory as provider_factory


def test_factory_threads_external_http_proxy_setting() -> None:
    config = app_config.AppConfig.from_env().with_overrides(
        llm_api_key=None,
        embedding_api_key=None,
        external_http_trust_env=False,
    )

    chat_provider = provider_factory.build_chat_provider(config)
    embedding_provider = provider_factory.build_embedding_provider(config)

    # Narrowing to the concrete types also asserts which providers the factory
    # chose.
    assert isinstance(chat_provider, chat.OpenAICompatibleChatProvider)
    assert isinstance(
        embedding_provider, embedding.OpenAICompatibleEmbeddingProvider
    )
    assert chat_provider._trust_env is False
    assert embedding_provider._trust_env is False
