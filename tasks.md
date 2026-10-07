# Current Tasks

This file is the short-horizon task board for the repository.

## Operating checklist

Before normal coding:

- Read root `AGENTS.md`.
- Skim only **Current focus** in this file.
- Open the one module `AGENTS.md` selected by the root router and at most one
  adjacent contract for a shared seam.
- Read `spec.md` only when the task needs product requirements. Read the
  roadmap only when the task may change phase status.
- Open a local skill when the task matches it.

A named, read-only repository audit may inventory multiple module contracts,
maintained documents, and implementation seams when that broad evidence is the
task. Historical snapshots under `docs/reviews/` remain evidence, not current
instructions, and are not rewritten during reconciliation.

After coding:

- Run the relevant module tests, then every applicable quality gate.
- Review the changed code and its module seams.
- Mark completed tasks below and add concrete follow-ups for new work.

## Current focus

Current active phase:

- Phase C **code complete, measurement not closed**
- `Phase D: Qdrant engineering` (lifecycle; not a substitute for eval gold)

Current theme:

- keep eval/compare honest (no fallback winners; gold is source substrings)
- make Qdrant indexing lifecycle stable and maintainable
- module contracts (`*/AGENTS.md`) are the parallel-development entry;
  root `AGENTS.md` is the router
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
  - Decision: **code complete, measurement not closed**. Do not claim local vs
    qdrant blended quality is comparable (`blended ≡ dense` on Qdrant).
- [ ] Optional: run broader retrieval tuning sweeps on a live provider once
      fallback cells stay off the leaderboard

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

### Audit remediation (2026-09-19)

- [x] Repair real Agents SDK tool registration when postponed annotations are
      resolved by the SDK decorator
- [x] Preserve safe SDK failure frame metadata without logging request content
      or provider credentials
- [x] Reject empty eval gold and exclude unmeasured comparison cells from the
      leaderboard
- [x] Keep retrieval and citation evidence matching within individual chunks
      and cited spans
- [x] Validate the full eval-comparison matrix before any cell can ingest
- [x] Exclude unknown provider or selected-reranker execution from comparison
      leaderboard eligibility
- [x] Audit Stage 1 after implementation: the combined review is clean after
      its two fix rounds; all gates, 185 tests, the frontend build, and a
      disposable no-key ingest/ask/eval/compare workflow pass
- [x] Bind versioned ingest manifests to canonical document scope and hashed
      storage identity, with conservative local/Qdrant legacy migration
- [x] Lock all backend extras and isolated build requirements; restore them
      with uv 0.12.16 in Python 3.11/3.13 CI, while preserving the frozen
      Node 22 and pnpm 10.18.0 frontend path
- [x] Close Stage 2 identity edge cases: preserve final index symlinks,
      normalize equivalent Qdrant targets, and sanitize malformed URLs without
      exposing embedded credentials
- [x] Audit Stage 2 after implementation: the combined review is clean after
      one fix round; locked gates and all 205 tests pass on Python 3.11/3.13,
      as do the Node 22 build, external wheel smoke, and no-key ingest/ask smoke
- [x] Serialize RAG index lifecycle operations on one host, publish durable
      repair intent, and add atomic disposable Qdrant initialization with
      fail-closed ownership cleanup
- [x] Isolate every eval-comparison cell from interactive storage, with stable
      dataset/configuration identities and exact-name Qdrant orphan metadata
- [x] Expose eval-comparison dataset/configuration identities and typed Qdrant
      orphan-recovery metadata through the HTTP response schema
- [x] Express index existence, incremental-update support, and live-embedding
      requirements through the `ChunkStore` lifecycle contract
- [x] Lock canonical manifest and storage resources together, recover Qdrant
      initialization interruptions, and sanitize outward Qdrant failures
- [x] Align Qdrant target identity with the locked client's effective request
      parser, including path normalization, strict IDNA, scoped IPv6, and
      authority-changing input rejection
- [x] Audit Stage 3 after implementation: the final combined review is Ready
      after its target-identity repair rounds; locked gates and all 385 tests
      pass on Python 3.11/3.13, as do the frozen frontend build, isolated wheel
      import/metadata check, and disposable no-key ingest/ask/eval/compare
      workflow with unchanged interactive storage
- [x] Preserve answer-wide provider status while merging repeated searches,
      assign citation labels within each answer, and expose stable source IDs
      through the HTTP response and frontend
- [x] Add frontend behavior coverage for answer-local citation labels,
      independent bootstrap failures, visible operation errors, retry state,
      and development/production API-origin selection; run it before the build
      in frozen Node 22 CI
- [x] Make API startup from an installed wheel independent of a source checkout,
      keep automatic static discovery constrained to the exact checkout, and
      fail fast for an invalid explicit frontend build
- [x] Add the single-host packaged deployment, backup, rollback, explicit static
      build, and offline rehearsal runbook without claiming multi-host Qdrant
      coordination
- [x] Reconcile every maintained instruction/reference with the hardened tree,
      HTTP wire, comparison defaults, generated artifacts, and live-operation
      risks; keep dated review snapshots historical
- [x] Enforce the complete backend dependency direction, delivery-framework
      isolation, the public RAG facade, and generic store capabilities with AST
      regression tests covering absolute and relative imports
  - Documentation/import close-out: 57 focused tests and 61 CLI/comparison
    tests pass. Three independent audit repair rounds closed root-facade and
    unknown-package checker escapes, generated build inventory and cache-cold
    wording gaps, and incomplete extension recipes. The final re-review is
    Ready with no remaining finding.
- [x] Audit Stage 4 after implementation: the combined review is Ready after
      `65bfcd0` closed one bootstrap-liveness finding and two Minor
      coverage/config gaps. At the repaired HEAD, locked Ruff, format, DOC201,
      mypy, all 449 tests, and compileall pass on Python 3.11/3.13; frozen Node
      22/pnpm passes all 15 frontend tests and the production build. Independent
      wheel, explicit-static, no-key local, import-boundary, and secret scans
      also pass. Live providers, live Qdrant, and Windows remain unverified.
- [x] Make the documented backend development command portable across Linux
      and Windows by launching through locked `uv`; verify it serves loopback
      health/info/documents on this Linux host. Keep PowerShell setup guidance.
- [ ] Resolve the upstream Starlette/AnyIO `BlockingPortal` deprecation warning
      or add a narrowly justified filter after the dependency lock is selected

### Structure and readability pass

- [x] Remove stray root shim scripts and untrack already-ignored build output
- [x] Rename `scripts/test_*.py` to `check_*` so manual checks are not mistaken for tests
- [x] Add `core/constants.py` as the single source of truth for closed option sets
- [x] Extract typed environment readers and validators into `core/env.py`
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

### Eval measurement (Phase A follow-through)

- [x] Add `retrieval_reciprocal_rank` and `retrieval_precision`, so the harness
      can see where evidence ranked rather than only whether it was retrieved
- [x] Rank the comparison leaderboard on them, replacing a primary key that
      rose with `top_k` and could not see reranking at all
- [x] Report "retrieved but not first" separately from "missed entirely"
- [x] Grow the eval set beyond two questions against one document (12 cases,
      multi-doc corpus under `data/corpus/sample/`; default
      `DOCS_DIR=data/corpus/sample`)
- [x] Write cases by failure mode (rare exact term, paraphrase, near-duplicate,
      cross-section, fusion definition). Hash lexical vs dense now disagree on
      mean RR; live-provider winners are still unrecorded
- [x] Require every `expected_retrieval_keywords` item to be a contiguous
      substring of its cited source (`tests/test_eval_gold.py`)
- [x] Compare aggregation: chat/embedding/runtime fallback and requested≠actual
      are `degraded`, excluded from the leaderboard; RR sort is tested
- [ ] Re-run `eval-compare` on a live provider once the set can discriminate,
      and record which strategy actually wins (do not rank fallback cells)
      - 2026-09-13 probe: embedding **live** (`text-embedding-v4`, 1024-d);
        chat **not live** (DashScope/compatible `Arrearage` / account not in
        good standing); Qdrant **skipped** (`WinError 10054` TLS reset).
        No leaderboard written — fallback cells must not rank.

### Agent-native modules

- [x] Root `AGENTS.md` is a module router; each package has `AGENTS.md`
- [x] Thin `core/` (`constants`, `models`, `exceptions`, `env`, `file_io`)
- [x] Production code outside `rag/` uses the rag facade
      (`tests/test_import_graph.py`)

### Ubuntu runner migration (2026-10-07)

- [x] Validate the existing CI gates on Ubuntu 26.04 without adding an OS
      matrix or changing toolchain versions, actions, dependencies, or locks.
  - Baseline: `3ec7f27`, [CI #14](https://github.com/gouyuanshuo/Local-Docs-RAG-Agent/actions/runs/36241580061)
    passed on Ubuntu 24.04.
  - Runner-only commit: `023de4e`,
    [CI #16](https://github.com/gouyuanshuo/Local-Docs-RAG-Agent/actions/runs/37597283922).
    All four jobs passed on Ubuntu 26.04.1, image `20260927.149.1`.
    Backend Python 3.11/3.13 each passed 449 tests and all existing gates;
    the real SDK regression passed 3 tests on each. Frontend passed 15 tests
    and its production build. The wheel built and served health outside the
    checkout after installation into a fresh environment.
  - Decision: pin all three job definitions to `ubuntu-26.04`. This uses the
    validated OS generation now and makes the next OS generation an explicit
    migration rather than a moving `ubuntu-latest` change. Versioned labels
    still receive image updates; this does not pin an exact image release.
    Keep the existing two Python entries and single frontend/package jobs.
  - uv caches missed on both backend jobs and the wheel job; the frontend
    reused the existing pnpm package cache. This is an equivalent workflow
    migration check, not a separate cache-cold frontend validation.
  - No system-package, prebuilt-binary, or Action failure occurred in these
    gates, so no compatibility repair or Ubuntu 24.04 fallback was needed.
    The existing Starlette/AnyIO deprecation and pnpm esbuild build-script
    warning remain; no unrelated dependency change was made.
  - Migration source: [runner-images #14748](https://github.com/actions/runner-images/issues/14748).
    Future compatibility failures can temporarily use `ubuntu-24.04` while
    a specific failing system package, binary, or Action is repaired.

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
