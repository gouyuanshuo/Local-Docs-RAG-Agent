# Module: runtime

How a question becomes an `AgentAnswer`. Delivery holds `LocalDocsAgent`
or calls `runtime.answer_question`.

## Public interface

- `runtime.answer_question(config, question)` — dispatcher
- `agent.LocalDocsAgent` — one-question facade (`agent.py` at package root)
- `tools.list_documents` / `search_documents` / `get_system_time` —
  primitives; `search_documents` returns `RetrievalOutcome` (pipeline +
  statuses)

Internals: `dispatch.py`, `basic.py`, `agents_sdk.py`, `shared.py`.

## Invariants

- Citation blocks: `[S1]` at column 0, `content:` last. Extractive chat
  fallback parses this; indentation lies and claims no context.
- Source ids are answer-local and first-seen by `chunk_id`. Basic and Agents
  SDK answers copy the exact ids shown to the model onto their citation spans;
  retrieval-owned hits and spans remain unchanged.
- `requested_runtime` vs `actual_runtime` always recorded.
- Agents SDK uses a request-owned client; never a process-global OpenAI
  client.
- Before decorating tools, the lazily imported `agents` module is bound in
  `agents_sdk` globals so the real SDK can resolve postponed annotations.
- SDK registration and run failures log only their class and innermost frame
  module, function, and line; they never render request data or exception text.
- Search goes through `rag.retrieve` (via `tools.search_documents`).
- Agents SDK search diagnostics aggregate across the whole run. Either
  fallback direction stays `fallback`; all distinct reasons, including live
  recovery reasons, retain first-seen order. SDK-to-basic fallback preserves
  prior tool-search degradation; a run with no search stays
  `unknown/search_not_run`.

## May change

- `agents_sdk` tool wiring, max turns, API style branch.
- Extra diagnostics fields **if** presenters + frontend types update too.

## Must not

- Put retrieval strategy into the prompt.
- Skip pipeline/rerank in the tool path.
- Import FastAPI / argparse.
- Indent the `[S*]` template or switch it to `textwrap.dedent`.

## Depends on / depended by

- Depends on: `rag` facade, `providers`, `config`, `core`.
- Depended by: delivery, evals (via `LocalDocsAgent`).

## Tests

`tests/test_runtime_shared.py`, `tests/test_agents_runtime.py`,
`tests/test_import_graph.py`.

## Parallel ownership

Own: `runtime/*`, `agent.py`, `tools.py`. Shared seam: `core.models`
answer/diagnostics shapes.

## New session

Root `AGENTS.md` + this file. Adjacent: `rag/AGENTS.md` only if changing
what `retrieve` returns.
