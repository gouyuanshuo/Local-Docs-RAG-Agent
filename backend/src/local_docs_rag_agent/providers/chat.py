from __future__ import annotations

import re
from textwrap import dedent

from openai import OpenAI

from local_docs_rag_agent.models import ProviderStatus
from local_docs_rag_agent.providers.base import ChatProvider
from local_docs_rag_agent.providers.errors import provider_error_reason
from local_docs_rag_agent.providers.openai_client import build_sync_openai_client

SYSTEM_INSTRUCTIONS = (
    "You are a local-document QA agent. "
    "Answer using only the provided context. "
    "If the context is insufficient, say what is missing instead of guessing. "
    "Keep the answer concise and cite source ids inline like [S1], [S2]."
)
CONTEXT_CONTENT_PATTERN = re.compile(r"(?ms)^content:\s*(.*?)(?=^\[S\d+\]\s*$|\Z)")


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
        self._provider_label = provider_label.lower()
        self._trust_env = trust_env
        self._status = ProviderStatus(provider=self._provider_label, mode="ready")
        self._client: OpenAI | None = None
        if api_key:
            self._client = build_sync_openai_client(
                api_key=api_key,
                base_url=base_url,
                trust_env=trust_env,
            )
        if not api_key:
            self._status = ProviderStatus(
                provider=self._provider_label,
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
            output_text = self._request_answer(prompt)
        except Exception as exc:
            self._status = ProviderStatus(
                provider=self._provider_label,
                mode="fallback",
                reason=provider_error_reason(exc),
            )
            return self._fallback_answer(question=question, context=context)

        if output_text:
            self._status = ProviderStatus(provider=self._provider_label, mode="live")
            return output_text

        self._status = ProviderStatus(
            provider=self._provider_label,
            mode="fallback",
            reason="empty_provider_output",
        )
        return self._fallback_answer(question=question, context=context)

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def _request_answer(self, prompt: str) -> str:
        if self._client is None:
            return ""
        if self._api_style == "chat_completions":
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            )
            return (completion.choices[0].message.content or "").strip()

        response = self._client.responses.create(model=self._model, input=prompt)
        return str(response.output_text).strip()

    def _fallback_answer(self, question: str, context: str) -> str:
        snippets = [
            match.group(1).strip()
            for match in CONTEXT_CONTENT_PATTERN.finditer(context)
            if match.group(1).strip()
        ]

        if not snippets:
            return f"I could not find supporting context for: {question}"
        excerpt = " ".join(snippets[:2])
        reason = self._status.reason or "live_provider_unavailable"
        return f"Provider fallback ({reason}); extractive answer: {excerpt}"
