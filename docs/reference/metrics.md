# Metrics

Every number an eval or a comparison reports: what it measures, how it is
computed, and what it cannot see. The last column matters as much as the
second. Several of these metrics are blind to exactly the change you might be
trying to measure.

All keyword matching is a case-insensitive substring test. An empty expectation
scores `1.0`, so a case can assert on some dimensions without being penalised
for the ones it leaves out.

## Per case

Each eval case produces one result with these metrics:

| Field | Measures | Blind to |
| --- | --- | --- |
| `retrieval_reciprocal_rank` | Mean over `expected_retrieval_keywords` of `1 / position`, where position is the first retrieved chunk containing the keyword. A keyword found nowhere scores `0`. `1.0` means every keyword was in the first result | whether the answer used the evidence |
| `retrieval_precision` | Share of retrieved chunks containing at least one expected retrieval keyword. `0` when something was expected and nothing was retrieved | where in the list the relevant chunks sit |
| `retrieval_span_hit_rate` | Share of `expected_retrieval_keywords` found anywhere in the retrieved chunks, joined into one text | order entirely; it also rises as `TOP_K` widens |
| `retrieval_source_hit_rate` | Share of `expected_source_paths` among the retrieved chunks' sources | order, and whether the right part of the source was retrieved |
| `answer_keyword_hit_rate` | Share of `expected_answer_keywords` found in the answer text | whether the answer is grounded or cited |
| `citation_source_hit_rate` | Share of `expected_source_paths` among the answer's cited sources | which spans were cited |
| `citation_span_hit_rate` | Share of `expected_span_keywords` found in the cited spans, joined into one text | order, and extra cited spans |
| `response_time_ms` | Wall time to produce the answer, rounded to 2 decimals | nothing it claims to measure, but it includes provider latency |

A result also carries the case's `question`, the `answer`, its `citations`, the
`retrieved_sources`, the answer's `diagnostics`, every expectation it was scored
against, and its `failure_reasons`.

### Why two retrieval metrics were added

`retrieval_span_hit_rate` joins the retrieved text before matching. It answers
"was the evidence retrieved at all" well, but reordering the same chunks cannot
change it, so a reranker scores exactly the same as no reranker. And because a
wider window has more text to match, a larger `TOP_K` can only raise it.
`retrieval_reciprocal_rank` sees position, and `retrieval_precision` makes a
padded window cost something. The same evidence at rank 1, 2, and 4 scores:

| Relevant chunk at | `retrieval_span_hit_rate` | `retrieval_reciprocal_rank` |
| --- | --- | --- |
| position 1 | 1.00 | 1.00 |
| position 2 | 1.00 | 0.50 |
| position 4 | 1.00 | 0.25 |

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

An empty run averages to `0.0`.

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

`/api/eval/compare` and `local-docs-rag eval-compare` return:

| Field | Value |
| --- | --- |
| `num_runs` | number of cells |
| one list per axis | the values swept: `runtimes`, `vector_backends`, `chunk_strategies`, `retrieval_strategies`, `rerankers`, `top_ks`, `chunk_sizes`, `chunk_overlaps` |
| `runs` | every cell |
| `leaderboard` | the cells that may be ranked, best first |

Each cell has a `label`, a `status`, a `reason` or `error` where one applies, its
`retrieval_config`, its `runtime`, and a `summary` when it ran. The `label` has
one segment per axis, in the order the axes are defined, for example
`basic:local:markdown:blended:rr-none:k4:s800:o120`.

A cell's `status` is `ok`, `degraded`, `skipped`, or `error`. What each means,
and the reasons attached to it, are in
[troubleshooting](../development/troubleshooting.md#the-comparison-matrix).

### Leaderboard

Only `ok` cells enter the leaderboard. A cell that ran on fallback measured hash
vectors or extractive answers rather than retrieval quality, so it is reported
but not ranked, and a comparison run without keys has an empty leaderboard.

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
