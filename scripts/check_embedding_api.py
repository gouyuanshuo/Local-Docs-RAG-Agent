"""Manual smoke check for the configured embedding provider.

Embeds one short string through the application's client construction path and
prints the resulting vector shape, which is the fastest way to confirm that the
configured model and dimension match what the vector store expects. This is not
a pytest module; run it directly with `python scripts/check_embedding_api.py`.
"""

from __future__ import annotations

import sys

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import exceptions
from local_docs_rag_agent.providers import openai_client

SAMPLE_TEXT = "Local Docs RAG Agent embedding connectivity check."


def main() -> int:
    _force_utf8_stdout()
    try:
        config = app_config.AppConfig.from_env()
        if not config.embedding_api_key:
            raise exceptions.ConfigurationError(
                "EMBEDDING_API_KEY (or a fallback LLM_API_KEY) must be "
                "set to run this check"
            )

        client = openai_client.build_sync_openai_client(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            trust_env=config.external_http_trust_env,
        )
        response = client.embeddings.create(
            model=config.embedding_model, input=[SAMPLE_TEXT]
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

    vector = response.data[0].embedding
    print(f"model={config.embedding_model}")
    print(f"text={SAMPLE_TEXT}")
    print(f"embedding_length={len(vector)}")
    print(f"embedding_preview={vector[:8]}")
    return 0


def _force_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
