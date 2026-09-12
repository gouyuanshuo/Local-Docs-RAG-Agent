# Module: core

Shared domain vocabulary and tiny I/O helpers. Nothing in this package may
import `config`, `rag`, `runtime`, `evals`, `providers`, or `api`.

## Public interface

| Symbol | File | Role |
| --- | --- | --- |
| option `Literal`s and tuples | `constants.py` | Single closed set for runtimes, chunk, backends, retrieval, rerankers |
| `DocumentChunk`, `RetrievalHit`, `AgentAnswer`, `Eval*` | `models.py` | Framework-free records |
| `LocalDocsError` and subclasses | `exceptions.py` | Expected failures with `code` / `action_hint` |
| `text`, `choice`, `load_project_dotenv`, … | `env.py` | Typed process-env readers |
| `atomic_write_text` | `file_io.py` | Sibling-tmp + fsync + replace |

Import: `from local_docs_rag_agent.core import constants` then
`constants.AGENT_RUNTIMES`. Do not import classes from `models` directly
(module-only import rule).

## Invariants

- Option lists are declared **once**, here. CLI, schema, config, and compare
  `AXES` derive from these tuples.
- `ConfigurationError` / `DataFormatError` map to HTTP 400;
  `ProviderUnavailableError` / `VectorStoreError` to 503 (mapping lives in
  delivery).
- Atomic write must not leave a truncated index/manifest.

## May change

- Adding a new closed option (and nothing else) is a core change: update
  `constants.py`, then the implementation that handles the new name, then
  callers' contracts.
- New fields on domain records **if** presenters and `frontend/src/types/api.ts`
  are updated in the same change (cross-module).

## Must not change without a named cross-module task

- Renaming `ProviderStatus.mode` values (`ready` / `live` / `fallback` /
  `unknown`).
- Dropping `action_hint` or stable `code` strings.
- Replacing atomic write with a non-atomic `Path.write_text`.

## Depends on / depended by

- Depends on: nothing in this package.
- Depended by: every other backend module.

## Tests

`tests/test_env.py`, `tests/test_config.py` (config uses core), and
`tests/test_import_graph.py` (core must not import siblings).

## Parallel ownership

Own: `core/*`. Do not edit `config/`, presenters, or frontend types unless
the task says the seam is moving.

## New session

Read root `AGENTS.md` and this file. Do not open `qdrant_store.py`.
