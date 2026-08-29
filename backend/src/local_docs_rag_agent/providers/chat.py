from __future__ import annotations

from textwrap import dedent

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.providers.base import ChatProvider


class OpenAICompatibleChatProvider(ChatProvider):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        api_style: str = "responses",
        provider_label: str = "LLM",
        trust_env: bool = True,
    ) -> None:
        self._model = model
        self._api_style = api_style
        self._provider_label = provider_label
        self._trust_env = trust_env
        self._status = ProviderStatus(provider=provider_label.lower(), mode="ready")
        self._client = self._build_client(api_key, base_url) if api_key else None
        if not api_key:
            self._status = ProviderStatus(
                provider=provider_label.lower(),
                mode="fallback",
                reason="missing_api_key",
            )

    def answer(self, question: str, context: str) -> str:
        if not self._client:
            return self._fallback_answer(question=question, context=context)

        prompt = dedent(
            f"""
            You are a local-document QA agent.
            Answer the user's question using only the provided context.
            If the context is insufficient, say what is missing instead of guessing.
            Keep the answer concise and cite source ids inline like [S1], [S2].

            Question:
            {question}

            Context:
            {context}
            """
        ).strip()

        try:
            if self._api_style == "chat_completions":
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a local-document QA agent. "
                                "Answer using only the provided context. "
                                "If the context is insufficient, say what is missing instead of guessing. "
                                "Keep the answer concise and cite source ids inline like [S1], [S2]."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                message = response.choices[0].message.content
                if message:
                    self._status = ProviderStatus(provider=self._provider_label.lower(), mode="live")
                    return message.strip()
                self._status = ProviderStatus(
                    provider=self._provider_label.lower(),
                    mode="fallback",
                    reason="empty_provider_output",
                )
                return self._fallback_answer(question=question, context=context)

            response = self._client.responses.create(model=self._model, input=prompt)
            output_text = response.output_text.strip()
            if output_text:
                self._status = ProviderStatus(provider=self._provider_label.lower(), mode="live")
                return output_text
            self._status = ProviderStatus(
                provider=self._provider_label.lower(),
                mode="fallback",
                reason="empty_provider_output",
            )
            return self._fallback_answer(question=question, context=context)
        except Exception as exc:
            self._status = ProviderStatus(
                provider=self._provider_label.lower(),
                mode="fallback",
                reason=f"provider_error:{exc.__class__.__name__}",
            )
            return self._fallback_answer(question=question, context=context)

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def _build_client(self, api_key: str, base_url: str | None):
        try:
            from openai import DefaultHttpxClient, OpenAI
        except Exception:
            self._status = ProviderStatus(
                provider=self._provider_label.lower(),
                mode="fallback",
                reason="openai_client_unavailable",
            )
            return None
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if not self._trust_env:
            client_kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
        return OpenAI(**client_kwargs)

    def _fallback_answer(self, question: str, context: str) -> str:
        snippets: list[str] = []
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line.startswith("content:"):
                continue
            snippet = line.removeprefix("content:").strip()
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return f"I could not find supporting context for: {question}"
        excerpt = " ".join(snippets[:2])
        return (
            f"{self._provider_label} API key is not configured or the provider client is unavailable, "
            f"so here is an extractive fallback: {excerpt}"
        )
