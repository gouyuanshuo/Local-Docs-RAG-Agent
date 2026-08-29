from __future__ import annotations

from openai import (
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    DefaultHttpxClient,
    OpenAI,
)


def build_sync_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    trust_env: bool,
) -> OpenAI:
    if trust_env:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=DefaultHttpxClient(trust_env=False),
    )


def build_async_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    trust_env: bool,
) -> AsyncOpenAI:
    if trust_env:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=DefaultAsyncHttpxClient(trust_env=False),
    )
