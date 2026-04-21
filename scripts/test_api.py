import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "qwen-plus")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你是谁？"},
            ],
        )
        print(completion.choices[0].message.content)
    except Exception as exc:
        print(f"错误信息: {exc}")
        print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")


if __name__ == "__main__":
    main()
