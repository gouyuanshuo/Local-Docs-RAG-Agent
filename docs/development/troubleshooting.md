# Troubleshooting

Start from the diagnostics, not the stack trace. The system reports degradation
as data rather than raising it, so most problems show up as a `fallback` mode
and a `reason` string long before they show up as an error.

## Reading an answer's diagnostics

Every answer, and every eval case, carries these fields:

| Field | Meaning |
| --- | --- |
| `requested_runtime` / `actual_runtime` | They differ when a runtime handed off to another |
| `vector_backend` | Which store served retrieval |
| `chat_provider` | The answering model's `provider`, `mode`, and `reason` |
| `embedding_provider` | The embedding provider's, for the query that was embedded |
| `reranker` | The second retrieval stage's |

Each status has one of four modes:

| Mode | Meaning |
| --- | --- |
| `live` | It really ran. A `reason` here is informational |
| `fallback` | It degraded, and `reason` says why. The result is still returned |
| `ready` | Configured but not exercised, such as a disabled reranker |
| `unknown` | Never reached, such as retrieval when the agent chose not to search |

A degraded answer is not an HTTP error: `/api/ask` returns `200` with the
fallback visible in the diagnostics, and the UI shows it as a warning.

## Provider reasons

| Reason | From | What happened | What to try |
| --- | --- | --- | --- |
| `missing_api_key` | chat, embeddings, reranker | No key is configured | Set the key in `.env` |
| `provider_error:<Exception>:<message>` | any provider | A call failed in a way retrying will not fix | Read the message: usually authentication, a wrong model name, or a wrong base URL |
| `provider_transient_error:<Exception>:retry_exhausted:<attempts>:max_retries:<n>` | embeddings | A retryable failure outlasted every retry | Retry later, or raise `EMBEDDING_MAX_RETRIES` and `EMBEDDING_RETRY_BACKOFF_MS` |
| `embedding_count_mismatch:…` | embeddings | The endpoint returned a different number of vectors than texts | Keep `EMBEDDING_BATCH_SIZE` within the endpoint's per-request limit |
| `empty_provider_output` | chat | The model returned no text | Check `LLM_MODEL` and `LLM_API_STYLE` |
| `unusable_rerank_output` | reranker | The reply held no usable JSON array of candidate numbers | Try another `RERANK_MODEL` |
| `recovered_after_retry:<n>` | embeddings, under `live` | It succeeded after retrying | Nothing; it is informational |
| `reranker_disabled` | reranker, under `ready` | `RERANKER=none` | Nothing, unless you meant to rerank |
| `rerank_not_needed` | reranker, under `ready` | One candidate, so no call was made | Nothing |
| `search_not_run` | agents runtime, under `unknown` | The agent answered without calling its search tool | Check whether the question needed documents |

## Runtime handoffs

When the `agents_sdk` runtime cannot serve a question, the basic runtime answers
it instead, `actual_runtime` becomes `basic`, and a `runtime_fallback:` reason
is recorded on the chat provider's status.

| Reason | Cause | Fix |
| --- | --- | --- |
| `runtime_fallback:agents_sdk_unavailable` | The `agents` extra is not installed | `python -m pip install -e ".[agents]"` |
| `runtime_fallback:missing_llm_api_key` | The Agents SDK needs a chat key | Set the chat key |
| `runtime_fallback:agents_sdk_error:<Exception>` | The agent run raised | Reproduce with the exception name; the basic answer is still returned |
| `runtime_fallback:empty_agent_output` | The agent produced no final output | Check the model and `AGENTS_MAX_TURNS` |

## HTTP errors

Expected failures map to status codes, and the body is always `code`, `detail`,
and `action_hint`:

| Status | Raised as | Typical cause |
| --- | --- | --- |
| `400` | `ConfigurationError`, `DataFormatError` | An invalid setting, a missing `DOCS_DIR`, a malformed index, manifest, or eval file |
| `503` | `ProviderUnavailableError`, `VectorStoreError` | A provider or Qdrant cannot do what was asked |
| `500` | any other expected failure | Report it; it should have a narrower type |

Read `action_hint` first. Every expected failure that has a reliable fix
carries one.

## Qdrant

**Nothing connects.** Setting `VECTOR_BACKEND=qdrant` without `QDRANT_URL` is a
configuration error. Once a URL is set, connection failures surface as a
`VectorStoreError`:

| `reason_code` | Meaning |
| --- | --- |
| `unreachable` | The service could not be reached |
| `operation_failed` | It was reached, and the operation failed |
| `dependency_missing` | The `qdrant` extra is not installed |
| `vector_size_mismatch` | The collection's dimension differs from the incoming embeddings |
| `invalid_vectors` | A chunk has no embedding, or the embeddings disagree in dimension |
| `invalid_collection` | The collection uses named vectors, or its size cannot be read |

`unreachable` is recognised from the error text, because the client wraps
transport errors from several libraries. It covers refused and reset
connections (including `WinError 10061` and `WinError 10054`), failed name
resolution, and a server that disconnected.

**A stale proxy.** Provider and Qdrant clients honour `HTTP_PROXY`,
`HTTPS_PROXY`, and `ALL_PROXY` by default. If a leftover proxy produces
connection errors that look like an outage, set `EXTERNAL_HTTP_TRUST_ENV=false`
in `.env` rather than editing the machine's environment. Keep it `true` when the
service is reachable only through a proxy. The unreachable hint tells you which
case you are in.

**Ingest or search refuses to run.** Qdrant rejects fallback embeddings, and
reports `Qdrant ingest requires a live embedding provider` or `Qdrant search
requires a live embedding provider`. Hash vectors are not comparable with the
live vectors already in a collection, so writing or querying with them would
corrupt results quietly. The action hint names the cause: a transient failure
suggests retries and backoff, anything else suggests checking
`EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, and `EMBEDDING_MODEL`. The local
backend has no such rule and stores fallback vectors.

**`blended` and `dense` rank identically.** On Qdrant, `blended` is the server's
own dense order, because the server returns no vectors to blend with. Compare
`dense` against `hybrid_rrf` there instead.

**A save failed halfway.** A Qdrant save deletes a changed document's old points
before writing the new ones. If the write fails, those sources are recorded in
the manifest's `needs_reindex`, and the next ask or eval re-ingests them even
though their checksums still match. Running `local-docs-rag ingest` does the
same immediately.

## The comparison matrix

Each cell of `eval-compare` ends in one of four statuses:

| Status | Meaning | On the leaderboard |
| --- | --- | --- |
| `ok` | Every case ran live on the requested runtime | yes |
| `degraded` | At least one case fell back: `runtime_fallback`, `chat_fallback`, `embedding_fallback`, or `reranker_fallback`, comma-separated | no |
| `skipped` | It could not run here: `missing_qdrant_url`, `qdrant_unreachable`, or `missing_qdrant_client` | no |
| `error` | It failed, as `<Exception>: <message>` | no |

An empty leaderboard after a keyless run is correct. A comparison of fallback
cells measures hash vectors and extractive answers, not retrieval quality, so
it has no winner.

A request expanding past 128 combinations, or naming an empty axis, is refused
as a configuration error before anything runs.

## The index and the corpus

**Results look wrong, or eval cannot find its sources.** Check `DOCS_DIR`. The
default is `data/corpus/sample`, and a `.env` value overrides it. Pointed at
`docs/`, it indexes the project's documentation instead of the corpus: the gold
set's sources are missing, and design notes about the eval topics become
distractors.

**The index predates a change.** The index fingerprint covers the vector
backend, the embedding provider, model, base URL, and dimensions, the embedding
mode, and the chunk strategy, size, and overlap. Changing any of them, including
adding or removing an embedding key, makes the next ask or eval rebuild the index.
A change the fingerprint does not cover, such as documents moving to new paths,
needs `local-docs-rag ingest`.

**`DOCS_DIR does not exist`.** The directory is resolved relative to where the
process starts. Run from the repository root, or use an absolute path.

## The environment

**`pydantic-core` fails to install or import on Windows.** `py -3` may have
chosen the free-threaded `3.13t` interpreter. Recreate the virtual environment
with `py -3.13`.

## Live checks

`scripts/` holds checks against real services. They are not tests: pytest never
collects them, they never run in CI, and their results are reported separately
from the offline gates.

| Script | Effect |
| --- | --- |
| `scripts/check_chat_api.py` | Read-only: one chat call |
| `scripts/check_embedding_api.py` | Read-only: one embedding call |
| `scripts/check_qdrant.py` | Read-only: reaches the configured collection |
| `scripts/verify_live_qdrant.py` | Writes to the configured collection, then asks and evaluates |

Run `verify_live_qdrant.py` only against a disposable or explicitly approved
collection. It fails if the runtime, chat, or embeddings degrade at any point,
which is what makes a pass meaningful.
