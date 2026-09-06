"""OpenAI-compatible chat provider with an explicit extractive fallback.

When no key is configured, or a request fails or returns nothing, the provider
answers by quoting the retrieved context instead of raising. The answer is
prefixed with the reason and the status is set to `fallback`, so a degraded
answer is always labelled as one rather than passing for a live model response.
"""

from __future__ import annotations

import re
import textwrap

import openai

from local_docs_rag_agent import models
from local_docs_rag_agent.providers import base as provider_base
from local_docs_rag_agent.providers import errors, openai_client

SYSTEM_INSTRUCTIONS = (
    "You are a local-document QA agent. "
    "Answer using only the provided context. "
    "If the context is insufficient, say what is missing instead of guessing. "
    "Keep the answer concise and cite source ids inline like [S1], [S2]."
)
CONTEXT_CONTENT_PATTERN = re.compile(
    r"(?ms)^content:\s*(.*?)(?=^\[S\d+\]\s*$|\Z)"
)


class OpenAICompatibleChatProvider(provider_base.ChatProvider):
    """Answers over an OpenAI-compatible endpoint, or extractively."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        api_style: str = "responses",
        provider_label: str = "LLM",
        trust_env: bool = True,
    ) -> None:
        """Build a chat provider over an OpenAI-compatible endpoint.

        A missing key is not an error: the provider answers extractively
        from the retrieved context and reports `fallback`, so a run
        without credentials still produces something inspectable.

        Args:
          api_key: Credential, or None to answer extractively.
          model: Model that answers.
          base_url: Endpoint override, or None for the OpenAI default.
          api_style: `responses` or `chat_completions`.
          provider_label: Name reported in diagnostics.
          trust_env: Whether to honour environment proxy variables.
        """
        self._model = model
        self._api_style = api_style
        self._provider_label = provider_label.lower()
        self._trust_env = trust_env
        self._status = models.ProviderStatus(
            provider=self._provider_label, mode="ready"
        )
        self._client: openai.OpenAI | None = None
        if api_key:
            self._client = openai_client.build_sync_openai_client(
                api_key=api_key,
                base_url=base_url,
                trust_env=trust_env,
            )
        if not api_key:
            self._status = models.ProviderStatus(
                provider=self._provider_label,
                mode="fallback",
                reason="missing_api_key",
            )

    def answer(self, question: str, context: str) -> str:
        """Answer `question` from `context`.

        Args:
          question: The user's question.
          context: Retrieved evidence, rendered as `[S1]` blocks.

        Returns:
          The model's answer, or an extractive one quoting the context
          when no live model is reachable. A degraded answer is prefixed
          with its reason and `status` reports `fallback`, so it can never
          pass for a live response. Never raises.
        """
        if not self._client:
            return self._fallback_answer(question=question, context=context)

        prompt = textwrap.dedent(
            f"""
            You are a local-document QA agent.
            Answer the user's question using only the provided context.
            If the context is insufficient, say what is missing
            instead of guessing.
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
            self._status = models.ProviderStatus(
                provider=self._provider_label,
                mode="fallback",
                reason=errors.provider_error_reason(exc),
            )
            return self._fallback_answer(question=question, context=context)

        if output_text:
            self._status = models.ProviderStatus(
                provider=self._provider_label, mode="live"
            )
            return output_text

        self._status = models.ProviderStatus(
            provider=self._provider_label,
            mode="fallback",
            reason="empty_provider_output",
        )
        return self._fallback_answer(question=question, context=context)

    @property
    def status(self) -> models.ProviderStatus:
        """Report whether the last answer was live or degraded."""
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

        response = self._client.responses.create(
            model=self._model, input=prompt
        )
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
