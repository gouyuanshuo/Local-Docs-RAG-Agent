import os
import sys

from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI

from local_docs_rag_agent.config import AppConfig


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    config = AppConfig.from_env()

    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    text = "通用多模态表征模型示例"

    if not api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY (or fallback LLM_API_KEY) in .env")

    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if not config.external_http_trust_env:
        client_kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
    client = OpenAI(**client_kwargs)
    response = client.embeddings.create(
        model=model,
        input=[text],
    )

    vector = response.data[0].embedding
    print(f"model={model}")
    print(f"text={text}")
    print(f"embedding_length={len(vector)}")
    print(f"embedding_preview={vector[:8]}")


if __name__ == "__main__":
    main()
