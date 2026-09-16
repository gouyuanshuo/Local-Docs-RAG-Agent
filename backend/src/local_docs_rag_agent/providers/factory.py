"""The single construction point for configured providers.

Everything else takes a built provider as an argument. That keeps `AppConfig`
reading in one place and makes providers straightforward to substitute in tests.
"""

from __future__ import annotations

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.providers import chat, embedding


def build_chat_provider(
    config: app_config.AppConfig,
) -> provider_base.ChatProvider:
    """Build the chat provider described by `config`.

    Args:
      config: The settings to read credentials and model from.

    Returns:
      A provider ready to answer. A missing API key is not an error
      here: the provider reports `fallback` and answers extractively.
    """
    return chat.OpenAICompatibleChatProvider(
        api_key=config.llm_api_key,
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_style=config.llm_api_style,
        provider_label=config.llm_provider.upper(),
        trust_env=config.external_http_trust_env,
    )


def build_embedding_provider(
    config: app_config.AppConfig,
) -> provider_base.EmbeddingProvider:
    """Build the embedding provider described by `config`.

    Args:
      config: The settings to read credentials, model, batching, and
        retry policy from.

    Returns:
      A provider ready to embed.
    """
    return embedding.OpenAICompatibleEmbeddingProvider(
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        batch_size=config.embedding_batch_size,
        max_retries=config.embedding_max_retries,
        retry_backoff_ms=config.embedding_retry_backoff_ms,
        provider_label=config.embedding_provider,
        trust_env=config.external_http_trust_env,
    )
