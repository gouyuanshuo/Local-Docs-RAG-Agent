"""Manual smoke check for the configured chat provider.

Sends one throwaway chat completion through the same client construction path
the application uses, so proxy and base-URL problems surface here instead of
mid-run. This is not a pytest module; run it directly with `python
scripts/check_chat_api.py`.
"""

from __future__ import annotations

import sys

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import exceptions
from local_docs_rag_agent.providers import openai_client

PROMPT = "Reply with a single short sentence confirming the connection works."


def main() -> int:
    _force_utf8_stdout()
    try:
        config = app_config.AppConfig.from_env()
        if not config.llm_api_key:
            raise exceptions.ConfigurationError(
                "LLM_API_KEY (or OPENAI_API_KEY) must be set to run this check"
            )

        client = openai_client.build_sync_openai_client(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            trust_env=config.external_http_trust_env,
        )
        completion = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
    except exceptions.LocalDocsError as exc:
        print(f"status=failed error={exc}", file=sys.stderr)
        return 1
    except (
        Exception
    ) as exc:  # A smoke check reports any provider failure, not just ours.
        print(
            f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1

    print(f"model={config.llm_model}")
    print(f"reply={(completion.choices[0].message.content or '').strip()}")
    return 0


def _force_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
