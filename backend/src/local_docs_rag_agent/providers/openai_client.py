"""Constructs OpenAI-compatible clients with an explicit proxy policy.

`trust_env` is threaded through deliberately: on a machine with stale
`HTTP_PROXY` or `HTTPS_PROXY` variables the default client fails with a
connection error that looks like a provider outage, so
`EXTERNAL_HTTP_TRUST_ENV=false` must be able to bypass the environment without
touching machine-wide settings.
"""

from __future__ import annotations

import openai


def build_sync_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    trust_env: bool,
) -> openai.OpenAI:
    """Build a synchronous OpenAI-compatible client.

    Args:
      api_key: Credential for the endpoint.
      base_url: Endpoint override, or None for the OpenAI default.
      trust_env: Whether to honour `HTTP_PROXY` and friends. Pass
        False to bypass a stale machine-wide proxy.

    Returns:
      The configured client.
    """
    if trust_env:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=openai.DefaultHttpxClient(trust_env=False),
    )


def build_async_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    trust_env: bool,
) -> openai.AsyncOpenAI:
    """Build an asynchronous OpenAI-compatible client.

    Args:
      api_key: Credential for the endpoint.
      base_url: Endpoint override, or None for the OpenAI default.
      trust_env: Whether to honour `HTTP_PROXY` and friends. Pass
        False to bypass a stale machine-wide proxy.

    Returns:
      The configured client. The caller owns it and must close it.
    """
    if trust_env:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=openai.DefaultAsyncHttpxClient(trust_env=False),
    )
