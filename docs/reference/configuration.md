# Configuration Reference

Every setting recognized by `AppConfig`: its type, default value, validation,
and what it changes.

Settings are read from the process environment and the project `.env` file (if
present). Copy `.env.example` to `.env` to configure a local instance.

---

## Shared Application Settings

| Variable | Type | Default | Validation / Values | Description |
| --- | --- | --- | --- | --- |
| `DOCS_DIR` | path | `data/corpus/sample` | readable directory | Directory holding source documents to ingest. |
| `DOCS_EXCLUDE_PATTERNS` | list[str] | `""` (empty) | comma-separated globs | Patterns under `DOCS_DIR` to exclude from ingestion. |
| `INDEX_PATH` | path | `data/index/chunks.jsonl` | writable path | File path for local JSONL vector/chunk store. |
| `INGEST_MANIFEST_PATH` | path | `data/index/ingest_manifest.json` | writable path | Manifest recording file checksums and configuration fingerprints. |
| `EVAL_PATH` | path | `data/evals/sample_eval.jsonl` | readable JSONL file | Evaluation dataset containing gold questions and retrieval expectations. |
| `AGENT_RUNTIME` | string | `basic` | `basic`, `agents_sdk` | Runtime orchestration engine to generate answers. |
| `AGENTS_MAX_TURNS` | integer | `6` | positive integer | Maximum execution steps for Agents SDK runtime. |
| `EXTERNAL_HTTP_TRUST_ENV` | boolean | `true` | `true`, `false`, `1`, `0` | Whether outbound HTTP clients trust system proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`). |

---

## Retrieval and Chunking

| Variable | Type | Default | Validation / Values | Description |
| --- | --- | --- | --- | --- |
| `VECTOR_BACKEND` | string | `local` | `local`, `qdrant` | Storage and similarity search backend for chunks. |
| `TOP_K` | integer | `4` | positive integer | Number of context chunks delivered to the answering prompt. |
| `CHUNK_STRATEGY` | string | `markdown` | `markdown`, `paragraph`, `fixed` | Strategy used to slice documents into chunks. |
| `CHUNK_SIZE` | integer | `800` | positive integer | Maximum character size per chunk. Must exceed `CHUNK_OVERLAP`. |
| `CHUNK_OVERLAP` | integer | `120` | non-negative integer | Overlap character count between consecutive chunks. |
| `RETRIEVAL_STRATEGY` | string | `blended` | `blended`, `dense`, `lexical`, `hybrid_rrf` | First-stage ranking algorithm. |
| `RETRIEVAL_CANDIDATE_K` | integer | `20` | positive integer | Number of candidates retrieved before rank fusion. Automatically widened if `TOP_K` is larger. |
| `RRF_K` | integer | `60` | positive integer | Damping constant for Reciprocal Rank Fusion. |
| `RERANKER` | string | `none` | `none`, `llm` | Optional second-stage reranking model. |
| `RERANK_CANDIDATE_K` | integer | `20` | positive integer | Number of candidate chunks fetched for reranking. |
| `RERANK_MODEL` | string | `None` | model identifier | Model used for LLM reranking; defaults to `LLM_MODEL` if omitted. |

---

## Language Model (LLM) Settings

| Variable | Type | Default | Validation / Values | Description |
| --- | --- | --- | --- | --- |
| `LLM_PROVIDER` | string | `openai` | provider identifier | LLM provider backend (e.g., `openai`, `qwen`). |
| `LLM_API_KEY` | string | `None` | API key token | Primary API key for LLM chat and answer generation. |
| `OPENAI_API_KEY` | string | `None` | API key token | Legacy fallback alias for `LLM_API_KEY`. |
| `LLM_BASE_URL` | string | `None` | HTTP(S) URL | Custom API base URL for OpenAI-compatible gateways. |
| `LLM_MODEL` | string | `gpt-4.1-mini` | non-empty string | Target model name used for question answering. |
| `OPENAI_MODEL` | string | `gpt-4.1-mini` | non-empty string | Legacy fallback alias for `LLM_MODEL`. |
| `LLM_API_STYLE` | string | `responses` | `responses`, `chat_completions` | Protocol dialect for OpenAI-compatible endpoints. |

---

## Embedding Provider Settings

| Variable | Type | Default | Validation / Values | Description |
| --- | --- | --- | --- | --- |
| `EMBEDDING_PROVIDER` | string | inherits `LLM_PROVIDER` | provider identifier | Provider for text embeddings (e.g., `openai`, `qwen`). |
| `EMBEDDING_API_KEY` | string | inherits `LLM_API_KEY` | API key token | API key for generating vector embeddings. |
| `EMBEDDING_BASE_URL` | string | inherits `LLM_BASE_URL` | HTTP(S) URL | Custom base URL for embedding endpoint. |
| `EMBEDDING_MODEL` | string | provider-dependent | non-empty string | Embedding model (e.g. `text-embedding-3-small`, `text-embedding-v4`). |
| `EMBEDDING_DIMENSIONS` | integer | `None` | positive integer | Optional explicit embedding dimension override. |
| `EMBEDDING_BATCH_SIZE` | integer | `10` | positive integer | Batch size for vector embedding requests. |
| `EMBEDDING_MAX_RETRIES` | integer | `2` | non-negative integer | Maximum retry attempts for failed embedding API calls. |
| `EMBEDDING_RETRY_BACKOFF_MS` | integer | `800` | non-negative integer | Initial exponential backoff in milliseconds between retries. |

---

## Qdrant Vector Database Settings

| Variable | Type | Default | Validation / Values | Description |
| --- | --- | --- | --- | --- |
| `QDRANT_URL` | string | `None` | HTTP(S) URL | Endpoint URL for remote or local Qdrant server. |
| `QDRANT_API_KEY` | string | `None` | API key token | Optional authentication token for Qdrant Cloud. |
| `QDRANT_COLLECTION` | string | `local-docs-rag` | non-empty string | Target collection name in Qdrant. |
| `QDRANT_TIMEOUT_S` | integer | `30` | positive integer | Request timeout in seconds for Qdrant client calls. |
