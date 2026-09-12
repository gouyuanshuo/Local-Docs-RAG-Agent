# Module: providers

Chat and embedding backends. Factory is the only construction point.
Status is data: a degraded call is `fallback` with a reason, not a fake
success and not an unlabelled exception (except Qdrant ingest, which
refuses hash vectors).

## Public interface

Import from `local_docs_rag_agent.providers`:

- `build_chat_provider(config) -> ChatProvider`
- `build_embedding_provider(config) -> EmbeddingProvider`

Protocols: `providers/base.py`. OpenAI-compatible implementations in
`chat.py` / `embedding.py`. HTTP clients: `openai_client.py` (`trust_env`).

## Invariants

- Missing API key → hash embeddings or extractive chat, `mode="fallback"`.
- Hash vectors are 128-d and **must not** be upserted or queried against a
  live Qdrant collection (enforced in rag ingest/search, not here).
- Do not assign `openai.OpenAI` / Agents SDK global clients.

## May change

- Retry / batching knobs already on `AppConfig`.
- A new OpenAI-compatible vendor is a factory branch plus defaults in
  config, not a new top-level package.

## Must not

- Construct clients outside the factory (except tests).
- Swallow errors into empty embeddings without setting `fallback`.
- Parse `[S1]` context with indented labels (that contract is runtime
  `shared.py`; chat fallback depends on it).

## Depends on / depended by

- Depends on: `config`, `core`.
- Depended by: rag (embed), runtime (chat), ingest.

## Tests

`tests/test_chat_provider.py`, `tests/test_embedding_provider.py`,
`tests/test_provider_factory.py`.

## Parallel ownership

Own: `providers/*`. Do not edit `runtime/shared.py` prompt layout from here
without a joint change.

## New session

Root `AGENTS.md` + this file.
