# Module: config

The immutable `AppConfig` snapshot every delivery path, ingest, runtime, and
eval cell reads. Variants come from `with_overrides`; construction always
re-validates.

## Public interface

| Symbol | Role |
| --- | --- |
| `AppConfig.from_env` | Load `.env` via `core.env`, never overriding exported vars |
| `AppConfig.with_overrides` / `with_runtime` | Frozen copy; unknown fields raise `ConfigurationError` |
| `AppConfig.__post_init__` | Invariants: overlap < size, positive k, backend URL when qdrant, … |

File: `config/__init__.py` (the package *is* the snapshot). Import as
`from local_docs_rag_agent import config as app_config`.

## Invariants

- Frozen + slots. No in-place mutation.
- `chunk_overlap >= chunk_size` is rejected here and again in the chunker.
- Default `docs_dir` is `data/corpus/sample`. The corpus lives outside
  `docs/`, so project documentation can never be indexed by default.

## May change

- New settings: add a field, read it with a `core.env` helper, validate in
  `__post_init__` (covers override variants). Then CLI/schema if the value
  is a public option — those option *names* still come from `core.constants`.

## Must not

- Split into sub-config objects unless that is the named task.
- Re-declare option literals (`"basic"`, `"blended"`, …).
- Treat missing API keys as a hard error at load time (providers fallback).

## Depends on / depended by

- Depends on: `core` only.
- Depended by: rag, providers, runtime, evals, delivery.

## Tests

`tests/test_config.py`, `tests/test_env.py`.

## Parallel ownership

Own: `config/`. Frozen unless the task is a new setting. Do not edit
`core/constants.py` from a config-only task without updating that contract.

## New session

Root `AGENTS.md` + this file. Adjacent: `core/AGENTS.md` if adding a closed
option.
