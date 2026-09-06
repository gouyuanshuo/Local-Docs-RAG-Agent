"""Reranks candidates by asking a chat model to order them against the question.

An LLM is the cheapest reranker to stand up that actually reads the question: no
second model to host, no extra dependency, and it works against any
OpenAI-compatible endpoint the project is already configured for. It is not the
cheapest to *run* — it costs one model call per question — which is why
`RERANKER=none` remains the default.

The model is asked for one thing and nothing else: a JSON array of candidate
numbers, best first. Ordering is a far narrower request than judging, so the
output is small, comparable across candidates, and cheap to validate. A number
that is out of range or repeated is dropped rather than trusted, and candidates
the model omitted are appended in their first-stage order, so the stage can
reorder the window but never lose part of it or fabricate a passage that
retrieval did not return.

Every failure lands in the same place: the candidates come back untouched and
the status becomes `fallback` with the reason attached. A reranker that raises
would turn a recoverable provider hiccup into a failed answer, and one that
silently returned the first-stage order would let a degraded run pass for a
reranked one.

After a successful rerank, `RetrievalHit.score` is a rank score — `1 / position`
— not a similarity. The model returns an ordering and no calibrated relevance,
so inventing one would be dishonest; as everywhere else in retrieval, scores are
comparable only within the strategy that produced them.
"""

from __future__ import annotations

import json
import re
import textwrap

import openai

from local_docs_rag_agent import models
from local_docs_rag_agent.providers import errors, openai_client
from local_docs_rag_agent.rag import scoring

SYSTEM_INSTRUCTIONS = (
    "You rank retrieved passages by how well they answer a question. "
    "Reply with a JSON array of passage numbers, best first, and nothing else."
)
# Candidates are truncated because the prompt holds the whole window at once: a
# window of twenty untruncated chunks is mostly prose the ranking decision never
# depends on.
CANDIDATE_TEXT_LIMIT = 600
JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]")


class LlmReranker:
    """Reorders a candidate window with one chat completion, degrading to identity."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        candidate_k: int,
        base_url: str | None = None,
        api_style: str = "responses",
        trust_env: bool = True,
    ) -> None:
        self._model = model
        self._candidate_k = candidate_k
        self._api_style = api_style
        self._client: openai.OpenAI | None = None
        self._status = models.ProviderStatus(provider="llm", mode="ready")
        if api_key:
            self._client = openai_client.build_sync_openai_client(
                api_key=api_key,
                base_url=base_url,
                trust_env=trust_env,
            )
        else:
            self._status = models.ProviderStatus(
                provider="llm",
                mode="fallback",
                reason="missing_api_key",
            )

    def candidate_depth(self, top_k: int) -> int:
        """Return the wider window this reranker reads before selecting `top_k`.

        Never narrower than `top_k`: a window smaller than the answer it feeds
        would discard results the caller asked for before the reranker ever saw
        them.
        """

        return max(self._candidate_k, top_k)

    def rerank(
        self, *, query: str, hits: list[models.RetrievalHit], top_k: int
    ) -> list[models.RetrievalHit]:
        if top_k <= 0 or not hits:
            return []
        if self._client is None:
            return hits[:top_k]
        if len(hits) == 1:
            # Nothing to reorder, so nothing worth paying a model call for.
            self._status = models.ProviderStatus(
                provider="llm",
                mode="ready",
                reason="rerank_not_needed",
            )
            return hits[:top_k]

        try:
            output_text = self._request_ranking(
                _build_prompt(query, hits, top_k)
            )
        except Exception as exc:
            self._status = models.ProviderStatus(
                provider="llm",
                mode="fallback",
                reason=errors.provider_error_reason(exc),
            )
            return hits[:top_k]

        order = _parse_ranking(output_text, len(hits))
        if not order:
            self._status = models.ProviderStatus(
                provider="llm",
                mode="fallback",
                reason="unusable_rerank_output",
            )
            return hits[:top_k]

        self._status = models.ProviderStatus(provider="llm", mode="live")
        return _apply_order(hits, order, top_k)

    @property
    def status(self) -> models.ProviderStatus:
        return self._status

    def _request_ranking(self, prompt: str) -> str:
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


def _build_prompt(
    query: str, hits: list[models.RetrievalHit], top_k: int
) -> str:
    # Candidates are numbered from 1 so the model never has to reason about
    # zero-based indexing, which it gets wrong far more often than it gets
    # ordering wrong.
    candidates = "\n\n".join(
        f"[{index}] {hit.chunk.title} ({hit.chunk.source_path})\n"
        f"{_preview(hit.chunk.text)}"
        for index, hit in enumerate(hits, start=1)
    )
    return textwrap.dedent(
        f"""
        Rank the passages below by how well each one answers the question.

        Question:
        {query}

        Passages:
        {candidates}

        Reply with a JSON array of at most {top_k} passage numbers,
        best first, and nothing else. Example: [3, 1]
        """
    ).strip()


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= CANDIDATE_TEXT_LIMIT:
        return collapsed
    return f"{collapsed[:CANDIDATE_TEXT_LIMIT].rstrip()}..."


def _parse_ranking(text: str, num_candidates: int) -> list[int]:
    """Return the zero-based candidate positions the model chose, in its order.

    Only the first JSON array in the reply is read, so a model that wraps it in
    a sentence of commentary is still understood. Anything that is not a usable
    candidate number is dropped; an empty result means the reply was unusable,
    which the caller reports as a fallback rather than as an ordering.
    """

    match = JSON_ARRAY_RE.search(text)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    order: list[int] = []
    for entry in parsed:
        number = _as_candidate_number(entry)
        if number is None or not 1 <= number <= num_candidates:
            continue
        position = number - 1
        if position in order:
            continue
        order.append(position)
    return order


def _as_candidate_number(entry: object) -> int | None:
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, str) and entry.strip().isdigit():
        return int(entry.strip())
    return None


def _apply_order(
    hits: list[models.RetrievalHit], order: list[int], top_k: int
) -> list[models.RetrievalHit]:
    chosen = set(order)
    reordered = [hits[position] for position in order]
    # A model that ranks only what it considers relevant leaves the rest in
    # first-stage order rather than dropping it, so a short reply cannot shrink
    # the answer window.
    reordered.extend(
        hit for index, hit in enumerate(hits) if index not in chosen
    )
    return [
        scoring.build_retrieval_hit(hit.chunk, 1.0 / position)
        for position, hit in enumerate(reordered[:top_k], start=1)
    ]
