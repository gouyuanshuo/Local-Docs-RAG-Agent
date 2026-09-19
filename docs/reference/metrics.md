# Metrics

Every number an eval or a comparison reports: what it measures, how it is
computed, and what it cannot see. The last column matters as much as the
second. Several of these metrics are blind to exactly the change you might be
trying to measure.

All keyword matching is case-insensitive substring matching. Answer keywords
are checked against the whole answer; retrieval and citation evidence is
checked within each individual chunk or span. An empty expectation scores
`1.0`, so a case can assert on some dimensions without being penalised for the
ones it leaves out.

## Per case

Each eval case produces one result with these metrics:

| Field | Measures | Blind to |
| --- | --- | --- |
| `retrieval_reciprocal_rank` | Mean over `expected_retrieval_keywords` of `1 / position`, where position is the first retrieved chunk containing the keyword. A keyword found nowhere scores `0`. `1.0` means every keyword was in the first result | whether the answer used the evidence |
| `retrieval_precision` | Share of retrieved chunks containing at least one expected retrieval keyword. `0` when something was expected and nothing was retrieved | where in the list the relevant chunks sit |
| `retrieval_span_hit_rate` | Share of `expected_retrieval_keywords` found within an individual retrieved chunk | order entirely; it also rises as `TOP_K` widens |
| `retrieval_source_hit_rate` | Share of `expected_source_paths` among the retrieved chunks' sources | order, and whether the right part of the source was retrieved |
| `answer_keyword_hit_rate` | Share of `expected_answer_keywords` found in the answer text | whether the answer is grounded or cited |
| `citation_source_hit_rate` | Share of `expected_source_paths` among the answer's cited sources | which spans were cited |
| `citation_span_hit_rate` | Share of `expected_span_keywords` found within an individual cited span | order, and extra cited spans |
| `response_time_ms` | Wall time to produce the answer, rounded to 2 decimals | nothing it claims to measure, but it includes provider latency |

A result also carries the case's `question`, the `answer`, its `citations`, the
`retrieved_sources`, the answer's `diagnostics`, every expectation it was scored
against, and its `failure_reasons`.

### Why two retrieval metrics were added

`retrieval_span_hit_rate` asks whether each evidence item appears within any
complete retrieved chunk. It cannot see rank, so reordering the same chunks
cannot change it, and a wider window can only raise it.
`retrieval_reciprocal_rank` sees position, and `retrieval_precision` makes a
padded window cost something. The same evidence at rank 1, 2, and 4 scores:

| Relevant chunk at | `retrieval_span_hit_rate` | `retrieval_reciprocal_rank` |
| --- | --- | --- |
| position 1 | 1.00 | 1.00 |
| position 2 | 1.00 | 0.50 |
| position 4 | 1.00 | 0.25 |

### Span metric semantics since 2026-09-19

Reports created before 2026-09-19 joined chunks and cited spans before testing
each evidence item. That could fabricate a hit across a chunk or span boundary.
Current reports require the complete item within one chunk or span. Do not
compare old and current span hit rates without labelling this semantics change.

## Failure reasons

`failure_reasons` names every dimension a case fell short on:

| Reason | When |
| --- | --- |
| `retrieval_missed_expected_source` | an expected source was not retrieved |
| `retrieval_missed_expected_span` | an expected retrieval keyword was not retrieved |
| `retrieval_ranked_expected_span_below_first` | every expected retrieval keyword was retrieved, but not all in the first result |
| `answer_missing_expected_keywords` | an expected answer keyword is missing from the answer |
| `citation_missed_expected_source` | an expected source was not cited |
| `citation_missed_expected_span` | an expected span keyword is missing from the cited spans |

"Retrieved but not first" is reported separately from "missed" because it is the
case a reranker exists to fix.

## Summary

`/api/eval`, `local-docs-rag eval`, and each comparison cell aggregate the cases:

| Field | Value |
| --- | --- |
| `num_cases` | number of cases scored |
| `runtime` | the runtime the run requested |
| `retrieval_config` | the settings the run used; see below |
| the seven rates above | each averaged over cases, rounded to 4 decimals |
| `avg_response_time_ms` | mean response time, rounded to 2 decimals |
| `results` | every per-case result |

Gold data must contain at least one case. The loader rejects an empty or
whitespace-only file before eval or comparison can initialize an index, run a
matrix cell, or emit a misleading all-zero summary.

Three older names remain for existing report consumers. Prefer the canonical
fields; the aliases carry the same values:

| Alias | Same value as |
| --- | --- |
| `avg_keyword_hit_rate` | `answer_keyword_hit_rate` |
| `source_hit_rate` | `citation_source_hit_rate`, not the retrieval source rate |
| `avg_citation_span_hit_rate` | `citation_span_hit_rate` |

`retrieval_config` records `vector_backend`, `chunk_strategy`, `chunk_size`,
`chunk_overlap`, `top_k`, `retrieval_strategy`, `retrieval_candidate_k`,
`rrf_k`, `reranker`, `rerank_candidate_k`, `docs_dir`, and
`docs_exclude_patterns`. It is what makes two runs comparable after the fact.

## Comparison report

`local-docs-rag eval-compare`, the internal comparison function, and
`/api/eval/compare` expose the same provenance-bearing report shape:

When axes are omitted, runtime, reranker, `top_k`, chunk size, and chunk overlap
hold their configured values; chunk strategy sweeps all three choices;
retrieval strategy sweeps all four; and Qdrant joins local only when a URL is
configured. The default is therefore 12 local cells or 24 with Qdrant.

| Field | Value |
| --- | --- |
| `num_runs` | number of cells |
| `dataset_identity` | stable digest of corpus relative paths/content checksums and gold bytes; it excludes document bodies and temporary storage |
| one list per axis | the values swept: `runtimes`, `vector_backends`, `chunk_strategies`, `retrieval_strategies`, `rerankers`, `top_ks`, `chunk_sizes`, `chunk_overlaps` |
| `runs` | every cell |
| `leaderboard` | the cells that may be ranked, best first |

Each cell repeats the `dataset_identity` and has a `label`, a `status`, a
`configuration_identity`, a `reason` or `error` where one applies, its
`retrieval_config`, its `runtime`, and a `summary` when it ran. The `label` has
one segment per axis, in the order the axes are defined, for example
`basic:local:markdown:blended:rr-none:k4:s800:o120`.

The raw report also gives every cell a `configuration_identity`, a digest of
the planned retrieval/runtime/provider-model settings. It intentionally omits
credentials, credential-bearing URLs, document bodies, and isolated storage
paths. A Qdrant cell records its exact disposable collection in
`run_metadata.disposable_qdrant_collection`; an unsuccessful cleanup also sets
`run_metadata.orphan_recovery_required`, `pre_cleanup_status`, and a cleanup
diagnostic. Its status becomes `error`, but its prior summary/reason remains in
the raw cell so the completed evaluation is not erased; the error cell is never
ranked. The CLI, raw writer, and typed FastAPI response preserve these
provenance fields. The typed API metadata also preserves `cleanup_error` and
`prior_error`, when present, so a cleanup failure does not erase the original
cell outcome.

A cell's `status` is `ok`, `degraded`, `skipped`, or `error`. What each means,
and the reasons attached to it, are in
[troubleshooting](../development/troubleshooting.md#the-comparison-matrix).

Before gold is loaded or a cell can touch storage, planning resolves the whole
Cartesian product, rejects empty axes and more than 128 cells, constructs every
validated `AppConfig` variant, and validates every executable storage target
through the RAG facade. A failure produces no partial run or CLI output. A
missing Qdrant URL remains the explicit `missing_qdrant_url` skipped cell; a
nonempty malformed Qdrant URL is a safe request-level configuration error.
Execution then isolates local cells in temporary files and Qdrant cells in
owned disposable collections. Configured live providers can make that sweep
billable; a local no-key run instead reports fallback cells and no winner.

### Leaderboard

Only `ok` cells enter the leaderboard. A cell that ran on fallback measured hash
vectors or extractive answers rather than retrieval quality, so it is reported
but not ranked, and a comparison run without keys has an empty leaderboard.
An empty result list or a result without diagnostics is likewise reported as
`degraded`, not ranked. Gold itself must contain at least one case; an empty
gold file is rejected before comparison ingestion begins. A disabled reranker
may report `ready`, which means the stage was not selected rather than verified
live execution. Chat and embedding must report `live`, as must a selected
reranker; otherwise the cell is degraded rather than ranked.

Each row repeats the cell's `label`, `retrieval_reciprocal_rank`,
`retrieval_precision`, `retrieval_span_hit_rate`, `retrieval_source_hit_rate`,
`answer_keyword_hit_rate`, `citation_span_hit_rate`, and `avg_response_time_ms`.
Rows are ordered by, in turn:

1. `retrieval_reciprocal_rank`, highest first
2. `retrieval_precision`, highest first
3. `answer_keyword_hit_rate`, highest first
4. `citation_span_hit_rate`, highest first
5. `avg_response_time_ms`, lowest first

## Choosing what to read

| Question | Read |
| --- | --- |
| Did retrieval find the evidence at all? | `retrieval_span_hit_rate`, `retrieval_source_hit_rate` |
| Did it rank the evidence first? | `retrieval_reciprocal_rank` |
| Is the window padded with noise? | `retrieval_precision` |
| Did the answer use the evidence? | `answer_keyword_hit_rate` |
| Did the answer cite the right place? | `citation_source_hit_rate`, `citation_span_hit_rate` |
