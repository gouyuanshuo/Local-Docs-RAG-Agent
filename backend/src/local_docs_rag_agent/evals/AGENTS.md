# Module: evals

Load gold, score one run, sweep a matrix, rank a leaderboard.

## Public interface

- `harness.load_eval_cases` / `run_eval`
- `harness.evidence_match_rate` / `reciprocal_rank` / `retrieval_precision`
- `harness.failure_reasons`
- `comparison.AXES` — single definition of sweepable axes
- `comparison.EvalMatrixPlan` / `plan_eval_matrix` / `run_eval_matrix`
- `comparison.cell_degradation_reason` / `run_label`
- Gold: `data/evals/sample_eval.jsonl`
- Corpus: `data/corpus/sample/` (default `DOCS_DIR`)

## Invariants

- Every `expected_retrieval_keywords` item is a **contiguous substring** of
  a cited source (`tests/test_eval_gold.py`).
- Retrieval and citation span evidence must occur within one retrieved chunk
  or cited span; it must not bridge their boundaries. Answer keyword scoring
  remains a whole-answer substring check.
- Gold JSONL must contain at least one case; empty or whitespace-only files
  fail before a comparison starts ingestion.
- Leaderboard sorts by `retrieval_reciprocal_rank` then precision, not by
  joined-string span hit rate.
- Chat/embedding/runtime fallback or `requested != actual` → cell
  `degraded`, **not** on the leaderboard. No-key/hash compare has no winner.
- Empty result lists and results without diagnostics are `degraded`, never
  leaderboard evidence. A disabled reranker may report `ready`; that means it
  was not selected, not that reranking ran live.
- Chat and embedding diagnostics must be `live` to rank. A selected reranker
  must also be `live`; only `ready` with `reranker_disabled` is rankable.
- Matrix cap 128; empty axis is an error; Qdrant unreachable is `skipped`.
- Matrix planning resolves every axis and validates every `AppConfig` variant
  before a cell can ingest. Defaults plan 12 local cells, or 24 with Qdrant.
- Every local matrix cell uses a fresh temporary index and manifest. Qdrant
  cells use an owned UUID-derived collection and temporary manifest; cleanup
  failures retain the exact disposable collection, cleanup diagnostics, and
  `pre_cleanup_status` in `run_metadata` while making the cell unrankable.
  A cleanup-origin interruption is re-raised with exact-name orphan recovery
  in an exception note, rather than replaced by a synthetic result.
- Raw comparison reports carry a stable corpus/gold `dataset_identity` and
  per-cell semantic `configuration_identity`; neither includes temporary
  storage paths, document bodies, keys, provider URLs, or resolved-away
  symlink aliases.
- `AXES` drives CLI flags, request schema field names, and labels.

## May change

- Add gold cases (keep 8–15 unless a named expansion). Failure-mode
  coverage: rare term, paraphrase, near-dup, cross-section, fusion.
- New compare axis: add one `MatrixAxis` in `AXES`, nothing else.

## Must not

- Rank fallback cells as `ok`.
- Re-introduce a position-coupled `product()` unpacking.
- Point `DOCS_DIR` at engineering `docs/*.md` by default.
- Import `api` or FastAPI.

## Depends on / depended by

- Depends on: `runtime` (`LocalDocsAgent`), `rag` facade, `presenters`,
  `config`, `core`.
- Depended by: delivery (`/eval`, `/eval/compare`, CLI).

## Tests

`tests/test_eval_gold.py`, `test_eval_harness.py`, `test_eval_comparison.py`.

## Parallel ownership

Own: `evals/*`, `data/evals/sample_eval.jsonl`, `data/corpus/sample/*`.
`presenters.py` is a shared seam with delivery.

## New session

Root `AGENTS.md` + this file. Adjacent: `api/AGENTS.md` if the compare
JSON shape changes.
