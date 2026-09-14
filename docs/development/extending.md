# Extending the system

Recipes for the extension points the design anticipates. Each one lists every
place a change has to reach. A missed step here rarely crashes: it shows up as a
setting the tests do not isolate, an option set declared twice, or a degraded run
that reads as a live one.

Paths are relative to `backend/src/local_docs_rag_agent/` unless they start at
the repository root.

Every recipe assumes the code style in [AGENTS.md](../../AGENTS.md#code-style):
module-only imports, 80 columns, and a Google docstring on anything public.
`ruff check` and `ruff format --check` enforce all three.

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
3. Define replacement and deletion explicitly. An incremental backend must
   honour `removed_source_paths` and `replaced_source_paths`, or points from a
   deleted or rewritten document survive in the index.
4. Normalize operational failures into `VectorStoreError` with a stable
   `reason_code`. The comparison matrix branches on it to report an unreachable
   service as a `skipped` cell rather than an `error`.
5. Construct it in `rag/store_factory.py` and export it from `rag/__init__.py`.
6. Add lifecycle and retrieval tests.

## A runtime

1. Add the name to `RuntimeName` in `core/constants.py`.
2. Return a complete `AgentAnswer` from `core/models.py`.
3. Record `requested_runtime` and `actual_runtime` truthfully. A runtime that
   hands off to another must say so, with a `runtime_fallback:` reason.
4. Retrieve through `rag.retrieve` and assemble citations with
   `runtime/shared.py`, so every runtime retrieves and cites the same way.
5. Wire it into `runtime/dispatch.py`.

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

## A comparison axis

1. Add the setting with the first recipe.
2. Add one `MatrixAxis` entry to `AXES` in `evals/comparison.py`, naming the
   request key, the `AppConfig` field it overrides, its CLI flag, its label
   prefix, and how it defaults.
3. Add the matching field, under the same name, to `EvalCompareRequest` and
   `EvalCompareResponse` in `api/schemas.py`.

Nothing else. The Cartesian product, the per-cell overrides, the run label, the
report keys, and the CLI flag are all derived from that entry, and
`tests/test_eval_comparison.py` fails if the schema, the CLI, or the label falls
out of step with it.

That derivation is the point. The axes were once a parameter list, a length
list, a `product()` call, and an unpacking tuple that had to agree by position.
Because every axis is a sequence, binding one axis's values to another's field
type-checked, ran, and produced quietly wrong numbers.

## A CLI command

1. Write a handler in `commands/`.
2. Add a `_register_*` function in `cli.py` that binds it with `set_defaults`.
3. List that function in `COMMAND_REGISTRARS`.

## After any extension

- If a public symbol changed, update that module's `AGENTS.md` in the same
  commit.
- Run the quality gates in [AGENTS.md](../../AGENTS.md#quality-gates).
