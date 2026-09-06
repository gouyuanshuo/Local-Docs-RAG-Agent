# Current Tasks

This file is the short-horizon task board for the repository.

## Operating checklist

Before coding:

- [x] Read `AGENTS.md`
- [x] Read `spec.md`
- [x] Read `tasks.md`
- [x] Check `docs/development-roadmap.md`
- [x] Open the relevant local skill if needed

After coding:

- [x] Run a relevant verification path
- [x] Review the changed code
- [x] Mark completed tasks below
- [x] Add follow-up tasks if new work appears

## Current focus

Current active phase:

- `Phase D: Qdrant engineering`

Current theme:

- make Qdrant indexing lifecycle stable and maintainable
- keep module boundaries and commit history legible as the surface grows

## Active tasks

### Phase C

- [x] Add configurable chunk strategies
- [x] Add document filtering and retrieval config snapshots
- [x] Add eval comparison matrix in CLI
- [x] Add compare API and frontend leaderboard
- [x] Add retrieval parameter comparison for `top_k`, `chunk_size`, and `chunk_overlap`
- [x] Run and analyze a stronger `local vs qdrant` comparison with a stable reachable Qdrant backend
- [x] Resolve current Qdrant connectivity issue blocking real compare runs (`qdrant_unreachable`)
- [x] Decide whether Phase C is complete enough to move primary focus to Phase D
- [ ] Optional: run broader retrieval tuning sweeps (more questions and larger docs) if needed

### Phase D preparation

- [x] Design incremental Qdrant ingest behavior
- [x] Define checksum/manifest strategy for document-level updates
- [x] Define stale document deletion behavior
- [x] Improve Qdrant ingest error reporting when embedding provider is fallback/transient
- [x] Add optional backoff/retry policy before failing Qdrant ingest on embedding instability
- [x] Normalize Qdrant network-unreachable errors into actionable ingest diagnostics
- [x] Add `source_path` payload index bootstrap for Qdrant delete filters
- [x] Add configurable Qdrant client timeout (`QDRANT_TIMEOUT_S`)
- [x] Add configurable embedding request batching (`EMBEDDING_BATCH_SIZE`)
- [x] Normalize proxy-env handling for external providers (`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`) in dev docs/bootstrap
- [x] Add focused automated tests for embedding batching/retry and incremental Qdrant lifecycle behavior
- [ ] Re-run live `qdrant ingest -> ask -> eval` from a network path that can reach the configured cloud endpoint
  - 2026-08-30: chat and embedding providers independently verified `live`
  - 2026-08-30: Qdrant TLS connection still resets with `WinError 10054` using the configured URL, disabled environment proxies, and explicit port `6333`
  - use `python scripts/verify_live_qdrant.py`; the gate rejects provider/runtime fallback

### Engineering hardening review (2026-08-29)

- [x] Create a clean Git baseline before the refactor (`3c9e342`)
- [x] Audit all Python source, test, and script files with Ruff, mypy, tests, and coverage
- [x] Validate configuration and persisted JSON at module boundaries
- [x] Split local store, Qdrant store, scoring, manifest, and atomic file I/O concerns
- [x] Fix missing-directory and changed-to-empty ingest consistency failures
- [x] Rebuild stale indexes when embedding or chunk configuration fingerprints change
- [x] Remove Qdrant result truncation and reject invalid/fallback vectors
- [x] Split FastAPI application assembly, routes, and typed schemas
- [x] Remove unused compatibility wrapper modules
- [x] Add architecture and deep-review documentation
- [x] Add strict Ruff/mypy development gates and regression tests
- [x] Add CI enforcement for Ruff, mypy, pytest, compileall, and frontend build
- [x] Split `frontend/src/App.tsx` into API, hooks, and feature components
- [x] Add deterministic Agents SDK fake-runner integration tests
- [x] Preserve runtime/provider diagnostics in per-case eval results
- [x] Add a strict live Qdrant verification script with stage-specific diagnostics

### Structure and readability pass

- [x] Remove stray root shim scripts and untrack already-ignored build output
- [x] Rename `scripts/test_*.py` to `check_*` so manual checks are not mistaken for tests
- [x] Add `constants.py` as the single source of truth for closed option sets
- [x] Extract typed environment readers and validators into `env.py`
- [x] Split the `ChunkStore` protocol, the JSONL store, and store construction apart
- [x] Move document discovery out of the ingest pipeline into `rag/discovery.py`
- [x] Promote the Qdrant error helpers to public names and drop the re-export shim
- [x] Add a `rag/__init__.py` facade and route external imports through it
- [x] Decompose `presenters.py` into composable serializers and remove the duplicate
      retrieval-config snapshot in `evals/comparison.py`
- [x] Replace the CLI command if-chain with an argparse handler registry
- [x] Fix the indented prompt-context blocks that silently disabled the extractive
      fallback, and cover the format contract with tests
- [x] Add module docstrings that record intent and invariants, not restated signatures
- [x] Rewrite the commit history into Conventional Commits

### Type-gate coverage

- [x] Annotate test fixtures and stubs so `mypy --strict` passes over `tests/`
- [x] Make test embedding stubs implement `EmbeddingProvider` instead of duck-typing it
- [x] Declare `files` in `[tool.mypy]` so a bare `mypy` checks source, tests, and scripts
- [x] Point the CI mypy step at the configured file set rather than `backend/src` alone

### Retrieval layering (Phase C depth)

- [x] Extract ranking out of the local store into `rag/retrieval.py`
- [x] Replace the bare lexical set-intersection with Okapi BM25 (`rag/bm25.py`)
- [x] Add reciprocal rank fusion (`rag/fusion.py`) instead of `max()` across scales
- [x] Make `RETRIEVAL_STRATEGY` a configuration setting and an eval-matrix axis
- [x] Re-rank Qdrant candidates client-side, since the server returns no vectors
- [x] Keep `blended` the default so earlier eval numbers stay reproducible
- [ ] Run a full `eval-compare` sweep over the strategies on a live provider and
      record which one wins on `retrieval_span_hit_rate`
- [x] Add a second-stage reranker behind an interface (LLM rerank first, leaving
      room for a cross-encoder)
- [x] Compose the two retrieval stages in one place (`rag/pipeline.py`) so a store
      stays unaware of reranking and the second stage cannot be skipped by accident
- [x] Carry the reranker's provider status into `AnswerDiagnostics`, so a degraded
      rerank is as visible as a degraded chat or embedding provider
- [ ] Add a cross-encoder reranker behind the same `Reranker` protocol, and compare it
      against `llm` on cost as well as on `retrieval_span_hit_rate`
- [ ] Measure how much of the rerank window is worth paying for: sweep
      `RERANK_CANDIDATE_K` once a real corpus sweep exists to compare against
- [ ] Add query rewriting / multi-query expansion ahead of candidate generation

### Google Python Style Guide conformance

- [x] Rewrite every symbol import as a module import (2.2), qualifying names at
      the use site
- [x] Rename the locals that shadowed a module after the import rewrite, which
      would have raised `UnboundLocalError` and which mypy does not report
- [x] Move test fakes from re-exported names to definition sites, which is what
      module-only imports leave to patch
- [x] Adopt the 80-column limit (3.2), holding code to it with `ruff format`
      and prose to it by hand
- [x] Document every public module, class, and function in Google style (3.8),
      with `Args:`, `Returns:`, and `Raises:` sections
- [x] Enforce it: `D` with the google convention in `pyproject.toml`, plus
      `ruff format --check` and a `DOC201` step in CI
- [x] Record the deliberate deviations and their reasons in `AGENTS.md`
- [ ] Reconsider `DOC501`/`DOC502` if ruff learns to see an exception raised by
      a helper or converted in place; until then `Raises:` is hand-maintained

### Structural follow-through (2026-09-07 review)

- [x] Drop the redundant `env_` prefix now that the module name supplies it
- [x] Replace the eval matrix's four position-coupled lists with one `AXES`
      table, and derive the product, overrides, label, report keys, and CLI
      flags from it
- [x] Cover the new coupling with tests: the request schema, the response
      schema, the CLI flags, and the run label must each name every axis
- [x] Untangle `_chunk_paragraphs` in `rag/chunker.py`: 51 statements and
      complexity 11, with five parallel mutable variables and three break
      conditions across two nested loops
- [x] Cover the chunker's grouping, section boundaries, and overlap advance
      with tests; it had four tests and none reached the paragraph path
- [ ] Consider splitting `rag/qdrant_store.py` (472 lines) along collection
      lifecycle, CRUD, search, and error normalization

### Follow-ups

- [ ] Consider grouping `AppConfig` into per-concern sub-configs if the field count
      keeps growing; the flat shape is still readable at its current size
- [ ] Consider promoting `data/` report writing behind a small reporting module if more
      report formats appear
- [ ] `LocalJsonlChunkStore.search` re-reads and re-parses the whole JSONL index on
      every query, and `rag/pipeline.retrieve` builds a fresh store per request, so an
      `eval-compare` sweep pays that cost once per question per cell. A cache keyed on
      the index path and its mtime would fix it, but chunks are mutable and shared, so
      it needs a considered ownership rule rather than a quick dictionary.

## Review notes

Use `skills/review-bugfix/SKILL.md` after meaningful code changes, especially when:

- runtime behavior changed
- retrieval logic changed
- eval output shape changed
- API payloads changed
