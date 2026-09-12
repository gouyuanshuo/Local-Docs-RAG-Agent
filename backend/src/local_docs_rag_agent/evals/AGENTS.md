# Module: evals

Load gold, score one run, sweep a matrix, rank a leaderboard.

## Public interface

- `harness.load_eval_cases` / `run_eval`
- `harness.reciprocal_rank` / `retrieval_precision` / `failure_reasons`
- `comparison.AXES` — single definition of sweepable axes
- `comparison.run_eval_matrix` / `cell_degradation_reason` / `run_label`
- Gold: `data/evals/sample_eval.jsonl`
- Corpus: `docs/sample/` (default `DOCS_DIR`)

## Invariants

- Every `expected_retrieval_keywords` item is a **contiguous substring** of
  a cited source (`tests/test_eval_gold.py`).
- Leaderboard sorts by `retrieval_reciprocal_rank` then precision, not by
  joined-string span hit rate.
- Chat/embedding/runtime fallback or `requested != actual` → cell
  `degraded`, **not** on the leaderboard. No-key/hash compare has no winner.
- Matrix cap 128; empty axis is an error; Qdrant unreachable is `skipped`.
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

Own: `evals/*`, `data/evals/sample_eval.jsonl`, `docs/sample/*`.
`presenters.py` is a shared seam with delivery.

## New session

Root `AGENTS.md` + this file. Adjacent: `api/AGENTS.md` if the compare
JSON shape changes.
