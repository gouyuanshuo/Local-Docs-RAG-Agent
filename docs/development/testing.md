# Testing

The suite is offline and deterministic. No test calls a live model, a live
embedding endpoint, or a Qdrant service, so a failure always means the code
changed, never that a network or a quota did.

## Running tests

- Everything: the pytest gate in [AGENTS.md](../../AGENTS.md#quality-gates).
- One module's tests: `python -m pytest tests/test_rerank.py -q`
- With coverage:
  `python -m pytest --cov=local_docs_rag_agent --cov-report=term-missing`

The frontend has no unit tests. Its gate is `pnpm run build`, which type-checks
with `tsc -b` before building.

## Where tests live

| Module | Tests |
| --- | --- |
| `core`, `config` | `tests/test_env.py`, `tests/test_config.py` |
| `providers` | `tests/test_chat_provider.py`, `tests/test_embedding_provider.py`, `tests/test_provider_factory.py` |
| `rag` | `tests/test_chunker.py`, `tests/test_ingest.py`, `tests/test_local_store.py`, `tests/test_qdrant_store.py`, `tests/test_retrieval.py`, `tests/test_rerank.py` |
| `runtime` | `tests/test_agents_runtime.py`, `tests/test_runtime_shared.py` |
| `evals` | `tests/test_eval_harness.py`, `tests/test_eval_comparison.py`, `tests/test_eval_gold.py` |
| delivery | `tests/test_api.py`, `tests/test_cli.py` |
| whole repository | `tests/test_import_graph.py`, `tests/test_docs.py` |

Each module's `AGENTS.md` lists the tests to run first when changing it.

## Environment isolation

`AppConfig.from_env` reads the process environment and loads a project `.env`.
Unchecked, a developer's own keys and endpoints would leak into every test, and
the suite would pass on one machine and fail in CI.

The autouse fixture in `tests/conftest.py` prevents that. It replaces the dotenv
loader with a no-op and deletes every variable in
`APPLICATION_ENVIRONMENT_VARIABLES`, so each test starts from defaults. A test
that needs a setting sets it explicitly with `monkeypatch.setenv`, where it reads
as part of the test.

A new configuration setting must be added to that tuple. See
[extending.md](extending.md#a-configuration-setting).

## Faking providers

Build the real provider with a fake key, then replace its client. The provider's
own batching, retry, parsing, and fallback logic still run; only the network is
gone.

- `_install_fake_endpoint` in `tests/test_embedding_provider.py` swaps
  `provider._client` for a stand-in whose `embeddings.create` returns, fails, or
  miscounts on demand.
- `tests/test_rerank.py` does the same for the reranker with
  `types.SimpleNamespace` objects exposing `responses.create` and
  `chat.completions.create`.

## Patch a name where it is defined

The codebase imports modules, not names, so a caller holds `store_factory` and
calls `store_factory.build_store`. There is no second binding to patch in the
caller's namespace. Patch the definition instead:

```python
monkeypatch.setattr(store_factory, "build_store", lambda config: fake_store)
```

That covers every caller at once, rather than only the one the test happened to
name.

## Tests that guard structure

Some tests assert nothing about behavior. They hold the repository's shape in
place, and a failure means a rule was broken rather than a feature.

- `tests/test_import_graph.py` enforces the one-way dependency direction and
  that code outside `rag` imports it only through its facade.
- `tests/test_eval_comparison.py` requires every `AXES` entry to reach the
  request schema, the response schema, a CLI flag, and the run label.
- `tests/test_eval_gold.py` requires gold keywords to be verbatim substrings of
  their sources and keeps the default corpus confined to `data/corpus/sample/`.
- `tests/test_docs.py` requires links and quoted paths to resolve and every
  document to be reachable from the index.

## Writing eval gold

Gold cases live in `data/evals/sample_eval.jsonl` and cite files under
`data/corpus/sample/`.

- Every item in `expected_retrieval_keywords` must appear, as a contiguous
  case-insensitive substring, in every file named in `expected_source_paths`.
  A keyword that appears nowhere caps the case's reciprocal rank below 1.0
  whatever retrieval does.
- Keep the set between 8 and 15 cases unless an expansion is the named task.
- Write each case to exercise a failure mode, so the set can separate the
  strategies: a rare exact term, a paraphrase with no shared terms, a
  near-duplicate distractor, an answer that crosses a section boundary, and a
  fact that only fusion ranks first.

## Refactoring without changing behavior

When no test reaches the code being restructured, write the evidence first.
Capture the code's output over real input before the change, and compare after.

The chunker split in `a878ec7` did this: it chunked the corpus under all three
strategies at four size and overlap settings, recorded every chunk's id,
offsets, title, metadata, and text hash, and required the 1800-chunk snapshot to
be identical afterwards.

## Proving a guard can fail

A new guard test that passes on its first run has not yet shown it guards
anything. Before trusting it, plant the defect it exists to catch, confirm the
test fails, and remove the plant. `tests/test_docs.py` was checked this way
against a stale quoted path, a broken link, and an unindexed document.

## Live checks are not tests

`scripts/check_*.py` and `scripts/verify_live_qdrant.py` talk to real services.
They are named so pytest never collects them and they never run in CI. See
[troubleshooting](troubleshooting.md#live-checks).
