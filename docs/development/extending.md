# Extending the system

Recipes for the extension points the design anticipates. Each one lists every
place a change has to reach. A missed step here rarely crashes: it shows up as a
setting the tests do not isolate, a browser wire union left stale, or a degraded
run that reads as a live one.

Paths are relative to `backend/src/local_docs_rag_agent/` unless they start at
the repository root.

Every recipe assumes the code style in [AGENTS.md](../../AGENTS.md#code-style):
module-only imports, 80 columns, and a Google docstring on anything public.
Ruff enforces import ordering, line length, docstrings, lint, and formatting.
Module-only imports are a code-review convention; the AST checks in
`tests/test_import_graph.py` enforce layer direction, use of the `rag` facade,
and capability-driven generic ingest lifecycle.

## A configuration setting

1. Add the field to `AppConfig` in `config/__init__.py`.
2. Read it in `AppConfig.from_env` with the matching reader from `core/env.py`.
3. Validate it in `AppConfig.__post_init__`. That also covers variants built by
   `with_overrides`, which never pass through the environment readers.
4. Add the variable to `.env.example`.
5. Add the variable to `APPLICATION_ENVIRONMENT_VARIABLES` in
   `tests/conftest.py`. Otherwise a developer's own `.env` supplies a value to
   every test that reads configuration, and the suite passes or fails for
   reasons that exist only on their machine.
6. If the setting is a closed set of options, declare it once in
   `core/constants.py` as a `Literal` alias with its derived tuple.
   Configuration parsing, request validation, and CLI choices all read it from
   there.
7. Add the setting, default, validation, and effect to
   `docs/reference/configuration.md`. `tests/test_docs.py` checks that every
   environment variable read by `AppConfig` is named there.
8. If the setting crosses the HTTP or browser boundary, update the matching
   `api/schemas.py` model, `frontend/src/types/api.ts`, UI control or display,
   behavior test, and HTTP reference. Run the frontend test and build gates;
   TypeScript option unions are hand-synchronized with Python.

## A provider

1. Implement `ChatProvider` or `EmbeddingProvider` from `providers/base.py`.
2. Report a truthful `ProviderStatus` through `status`: `live` only when a real
   call succeeded, and `fallback` with a `reason` whenever the result is
   degraded. Do not raise for a recoverable provider fault, and never return a
   degraded result as `live`.
3. Construct it in `providers/factory.py`. Nothing else builds providers.
4. Add its settings with the recipe above, and test it with a fake endpoint as
   described in the [testing guide](testing.md#faking-providers).

## A vector store

1. Add the name to `VectorBackendName` in `core/constants.py`.
2. Implement `ChunkStore` from `rag/base.py` in its own module.
3. Declare the lifecycle capabilities truthfully:
   - `exists()` reports whether this exact configured index target exists. It
     must not report the availability of a different file, collection, or
     service-wide resource.
   - `supports_incremental_updates` is `True` only when `save` can preserve
     unchanged sources while applying source-level deletion and replacement.
     A `False` store receives the complete corpus and rewrites it.
   - `requires_live_embeddings` is `True` when fallback vectors are unsafe for
     the store. Ingest then rejects them before publishing a store mutation.
4. Define replacement and deletion explicitly. An incremental backend must
   honour both `removed_source_paths` and `replaced_source_paths`, including a
   replacement that produces no chunks, or stale records survive. Those lists
   are `None` for a full-replacement backend.
5. Normalize operational failures into `VectorStoreError` with a stable
   `reason_code`. The comparison matrix branches on it to report an unreachable
   service as a `skipped` cell rather than an `error`.
6. Extend `rag/ingest.validate_storage_target` and storage identity together.
   Pure validation must not inspect files, acquire locks, build a client, contact
   a service, or write anything. Give the backend an explicit identity branch,
   canonicalize only non-secret target data, strip credentials, and never let a
   new backend fall through to another backend's identity. An unsupported or
   incompletely wired backend must fail closed before it can claim or mutate
   storage.
7. Define the backend's legacy-manifest ownership policy in
   `_validate_manifest_storage`. Decide deliberately whether an identity-less
   legacy manifest can rebuild and adopt this target, or must require an
   explicit operator ownership decision; do not inherit another backend's
   policy by fallthrough.
8. Add a matching storage resource in `rag/index_lock.py`. The lifecycle guard
   always locks both the bound manifest and store target in global key order.
   Use the same canonical target as validation, identity, and I/O; put local
   lock artifacts beside the resolved target or remote artifacts in a per-user
   cache, and keep credentials out of the resource key and persistent artifact.
9. Construct it in `rag/store_factory.py` and export it from `rag/__init__.py`.
   Do not add backend-name or concrete-type branches to generic ingest or
   readiness code; factory selection, storage identity, legacy ownership, and
   lock mapping are the deliberately backend-specific exceptions.
10. Decide how eval comparison isolates this backend before exposing it as an
    axis choice. Local cells use temporary files; Qdrant cells use the
    ownership-token initialization/deletion facade and UUID collection names.
    A different persistent service needs an equally fail-closed, exact-target
    ownership and cleanup path; it must not run through the local comparison
    branch and mutate an interactive target.
11. Add lifecycle tests for missing, unchanged, changed, removed, and
    changed-to-empty sources, plus retrieval and normalized-failure tests. Add
    identity/lock tests proving equivalent forms of the same target
    intentionally collide, distinct backends or targets do not collide,
    secrets never enter persisted identity/lock data, and unsupported or
    unwired backend names fail closed. For a remote comparison backend, test
    absence checks, ownership-token rejection, cleanup failure metadata, and
    exact-name orphan recovery.
12. Update `VectorBackendName` in `frontend/src/types/api.ts`; it is a
    hand-synchronized copy of the Python option set used by app-info,
    diagnostics, retrieval configuration, and comparison reports. Run the
    frontend behavior tests and build. Also update the vector-backend values
    in `docs/reference/configuration.md`, plus the HTTP, CLI, deployment,
    backup, and troubleshooting guidance affected by the backend's operating
    model.

## A runtime

1. Add the name to `RuntimeName` in `core/constants.py`.
2. Return a complete `AgentAnswer` from `core/models.py`.
3. Record `requested_runtime` and `actual_runtime` truthfully. A runtime that
   hands off to another must say so, with a `runtime_fallback:` reason.
4. Retrieve through `rag.retrieve` and assemble citations with
   `runtime/shared.py`, so every runtime retrieves and cites the same way.
5. Wire it into `runtime/dispatch.py`.
6. Add the name to the hand-synchronized `RuntimeName` union in
   `frontend/src/types/api.ts` and add its explicit selector option in
   `frontend/src/components/AskPanel.tsx`. Cover selection and response
   rendering in a frontend behavior test, then run the test and build gates.
7. Update the runtime values and behavior in
   `docs/reference/configuration.md`, `docs/reference/http-api.md`, and the
   relevant CLI/troubleshooting guidance.

## A reranker

1. Add the name to `RerankerName` in `core/constants.py`. Configuration, request
   validation, CLI choices, and the comparison axis pick it up from there.
2. Implement the `Reranker` protocol from `rag/rerank.py` in its own module. Take
   no dependency on which store produced the candidates.
3. On every failure, return the candidates unchanged and report `fallback`. A
   reranker must not raise: a recoverable provider fault would otherwise become
   a failed answer, and a silent retreat to first-stage order would let a
   degraded run pass for a reranked one.
4. Say in `candidate_depth` how wide a window it reads, and never read less than
   `top_k`.
5. Construct it in `build_reranker` in `rag/pipeline.py`.
6. Test that it reorders, that a bad reply degrades rather than raises, and that
   its status reaches the answer diagnostics.
7. Add the name to the hand-synchronized `RerankerName` union in
   `frontend/src/types/api.ts`; update a browser control and its behavior test
   if the UI exposes selection. Run the frontend tests/build, and update the
   allowed values and behavior in `docs/reference/configuration.md`, the CLI
   reference, HTTP/report examples, and metrics guidance.

## A retrieval strategy

1. Add the name to `RetrievalStrategyName` in `core/constants.py`.
2. Implement the ranking branch in `rag/retrieval.py`, returning hits built by
   `build_retrieval_hit` from `rag/scoring.py` so citation spans stay attached.
3. Decide what the strategy means on Qdrant, which returns payloads without
   vectors. Either handle it in `rerank_dense_hits`, or declare in
   `needs_candidate_window` that it needs no wider window.
4. Add tests showing it ranks *differently* from an existing strategy, not
   merely that it runs. A strategy nothing can distinguish is not a strategy.
5. Sweep it against `blended` with `eval-compare` before claiming it is better,
   using live providers. Cells that ran on fallback are `degraded` and stay off
   the leaderboard, so a keyless sweep has no winner.
6. Add the name to the hand-synchronized `RetrievalStrategyName` union in
   `frontend/src/types/api.ts`; update a browser control and behavior test if
   selection is exposed. Run the frontend tests/build, and update the allowed
   values and semantics in `docs/reference/configuration.md`, the CLI and HTTP
   references, and `docs/reference/metrics.md`. Because the matrix default
   sweeps every retrieval strategy, adding one changes the bare comparison's
   cell count, provider cost, and remote-mutation count; update every stated
   default count rather than leaving the current 12/24 figures in place.

## A comparison axis

1. Add the `AppConfig` setting with the first recipe, including
   `.env.example`, test-environment isolation, validation, and the
   configuration reference.
2. Add one `MatrixAxis` entry to `AXES` in `evals/comparison.py`. Its `name` is
   the request and report key; `field` is the real `AppConfig` override;
   `flag`, `noun`, and `label_prefix` drive CLI/help/labels; `choices` provides
   a closed option set or `None` for integers; and `default` returns the values
   used when the caller omits the axis. Decide that default deliberately. A
   sweep can multiply calls and cost; holding steady uses the configured value.
3. Add a constrained list alias when needed and a field under exactly
   `axis.name` to both `EvalCompareRequest` and `EvalCompareResponse` in
   `api/schemas.py`. The response field is required because every report echoes
   its resolved axes; the request field is optional because omission selects
   the axis default.
4. The CLI parser derives repeatable flags from `AXES`, so do not add a second
   hand-written flag. Confirm the generated flag's choices/type and destination
   in `tests/test_eval_comparison.py`, and document it in
   `docs/reference/cli.md`, including what omission does.
5. Add the resolved response field to `CompareResponse` in
   `frontend/src/types/api.ts`. If the UI can request values, add that request
   field to its typed payload/control and to the JSON body in
   `frontend/src/hooks/useRagWorkspace.ts`; cover the interaction with a
   behavior test. Run both the frontend test and build gates so the
   hand-synchronized wire type is checked.
6. Update `docs/reference/http-api.md` request and response examples,
   `docs/reference/metrics.md` report fields and label description,
   `docs/reference/cli.md` flags/default count, and
   `docs/reference/configuration.md` for the underlying setting. If the new
   default changes cell count, cost, or remote mutations, say so explicitly in
   the CLI and HTTP references.
7. Run `tests/test_eval_comparison.py`, `tests/test_cli.py`,
   `tests/test_docs.py`, the frontend behavior tests/build, and then the
   applicable root quality gates.

No other matrix-planning code changes are needed. The Cartesian product, the
per-cell overrides, the run label, the report keys, and the CLI flag are all
derived from that entry, and `tests/test_eval_comparison.py` fails if the
Python schema, the CLI, or the label falls out of step with it. HTTP examples,
TypeScript wire types, UI request serialization, and prose references remain
explicit delivery-boundary updates rather than derived artifacts.

That derivation is the point. The axes were once a parameter list, a length
list, a `product()` call, and an unpacking tuple that had to agree by position.
Because every axis is a sequence, binding one axis's values to another's field
type-checked, ran, and produced quietly wrong numbers.

## A CLI command

1. Write a handler in `commands/`.
2. Import that command module in `cli.py`, using the documented module alias
   convention when one exists.
3. Add a `_register_*` function in `cli.py` that binds the handler with
   `set_defaults`.
4. List that function in `COMMAND_REGISTRARS`.
5. Add parser/flag/dispatch coverage in `tests/test_cli.py` and document the
   command and every public flag in `docs/reference/cli.md`.
   `tests/test_docs.py` checks that the reference covers the generated CLI
   surface.

## After any extension

- If a public symbol changed, update that module's `AGENTS.md` in the same
  commit.
- Run the quality gates in [AGENTS.md](../../AGENTS.md#quality-gates).
