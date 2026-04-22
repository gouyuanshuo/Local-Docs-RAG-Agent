# Local Docs RAG Agent Development Roadmap

This file is the working development plan for the project.

It exists to keep the project moving in a consistent direction:

- keep the architecture clear
- keep priorities stable
- avoid random feature creep
- make each next step understandable

When continuing development, use this file as the default roadmap and decision reference.

---

## 1. Project identity

**Project name:** Local Docs RAG Agent

**Core idea:**

This is not a model-training project.
It is a small AI application system built around:

`Agent -> Tools -> RAG -> Citation -> Eval`

The goal is to let an LLM do useful work around private/local documents:

- read local files
- retrieve relevant knowledge
- answer with citations
- expose tool-using behavior
- evaluate results
- remain extensible across model providers

---

## 2. Current architecture

### Frontend

- `frontend/`
- stack: `Vite + React + TypeScript`
- responsibilities:
  - ask questions
  - trigger ingest/eval
  - show answer cards
  - show citation spans
  - show system/config state

### Backend

- `backend/src/local_docs_rag_agent/`
- stack: `FastAPI + Python package`
- responsibilities:
  - config loading
  - provider setup
  - ingestion
  - retrieval
  - runtime dispatch
  - eval execution
  - API responses

### Main backend modules

- `api/`
  - HTTP endpoints and schemas
- `commands/`
  - CLI entrypoints
- `providers/`
  - chat provider
  - embedding provider
  - provider factory
- `rag/`
  - chunking
  - embedding flow
  - local store
  - Qdrant store
- `runtime/`
  - `basic`
  - `agents_sdk`
  - shared retrieval/answer assembly
- `evals/`
  - eval harness

### Current supported capabilities

- local docs ingest
- local retrieval
- Qdrant retrieval
- citation span tracking
- `basic` runtime
- `agents_sdk` runtime
- eval harness
- OpenAI-compatible providers
- Qwen via compatible endpoint
- split frontend/backend dev workflow

---

## 3. Development principle

Do not prioritize adding more modules first.

Use this order:

1. measurable
2. explainable
3. reliable
4. more agent-like
5. more product-like

In short:

**first make the system understandable, then make it stronger.**

---

## 4. Priority roadmap

## Phase A: Eval reinforcement

**Goal:** upgrade eval from a smoke test into a system that genuinely guides iteration.

**Priority:** highest

### What to improve

- upgrade eval case schema
  - `expected_answer_keywords`
  - `expected_source_paths`
  - `expected_span_keywords`
  - `expected_retrieval_keywords`
  - `notes`
- split metrics
  - `retrieval_source_hit_rate`
  - `retrieval_span_hit_rate`
  - `answer_keyword_hit_rate`
  - `citation_source_hit_rate`
  - `citation_span_hit_rate`
  - `response_time_ms`
- output per-case diagnostics
  - expected vs retrieved
  - expected vs cited
  - failure classification
- support configuration comparison
  - `basic vs agents_sdk`
  - `local vs qdrant`

### Why this comes first

The most important problem is no longer:

“Can the system run?”

It is:

“How do we know where it is failing?”

### Current status

- first version already implemented
- next step: compare configurations and expose richer reporting in the frontend

---

## Phase B: Explicit degradation and reliability

**Goal:** replace silent fallback behavior with visible, structured system state.

**Priority:** very high

### What to improve

- explicit chat provider fallback status
  - e.g. `provider_status=degraded`
  - `answer_mode=fallback`
- explicit embedding provider fallback status
  - e.g. `embedding_mode=hash_fallback`
- explicit `agents_sdk -> basic` fallback status
- include actual runtime/provider state in API responses
- show system state in frontend
  - live provider
  - fallback mode
  - runtime actually used

### Why this comes second

If fallback behavior is invisible, eval results and optimization decisions become misleading.

### Current status

- first version already implemented
- answer diagnostics now expose:
  - requested runtime
  - actual runtime
  - chat provider mode/reason
  - embedding provider mode/reason
- next step: thread this more clearly into eval reports and status dashboards

---

## Phase C: Retrieval engineering reinforcement

**Goal:** move RAG from “working” toward “retrieval system quality”.

**Priority:** high

### What to improve

- chunk strategy experiments
  - fixed character chunks
  - paragraph-based chunks
  - heading/hierarchy-aware chunks
- retrieval comparison framework
  - local hybrid
  - Qdrant dense
- metadata improvements
  - source title
  - section title
  - chunk hierarchy
- bring retrieval settings into eval comparison
  - top-k
  - chunk size
  - overlap

### Why this matters

Retrieval engineering is one of the project’s main sources of portfolio value.

### Current status

- configurable chunk strategies are implemented:
  - `fixed`
  - `paragraph`
  - `markdown`
- document filtering and retrieval-config snapshots are implemented
- eval comparison matrix is implemented in CLI
- compare API and frontend leaderboard are implemented
- retrieval parameter comparison is implemented for:
  - `top_k`
  - `chunk_size`
  - `chunk_overlap`
- next step: run and analyze stronger `local vs qdrant` comparisons, then decide whether to close Phase C or continue deeper retrieval tuning

---

## Phase D: Qdrant engineering

**Goal:** move vector storage from “demo-style rebuild” to sustainable indexing behavior.

**Priority:** medium-high

### What to improve

- replace destructive full rebuild with incremental upsert
- use `chunk_id` / checksum for document-level updates
- support deletion of stale documents
- keep ingest manifest/checkpoint data
- make collection/index readiness explicit

### Why this is not earlier

It matters, but measurement and reliability improve project credibility more directly.

---

## Phase E: Agentization upgrades

**Goal:** make `agents_sdk` more agent-like, not just tool-capable.

**Priority:** medium

### What to improve

- retry / reformulate when retrieval is weak
- answer self-check
- reviewer pass
- tool call logging / traces
- clearer reasoning-step observability

### Why this is not first

Agent behavior becomes harder to improve if eval and reliability foundations are weak.

---

## Phase F: Product experience round two

**Goal:** move from “usable frontend” to “more complete application UX”.

**Priority:** medium

### What to improve

- eval result visualization
- question / answer history
- clickable citation expansion / highlighting
- document upload / rebuild entrypoint
- clearer system state page
- configuration switch panel

### Why this is later

The frontend already has a first usable version.
The next highest-value work is still below the UI layer.

---

## Phase G: Provider expansion

**Goal:** fully demonstrate backend switchability across model providers.

**Priority:** medium-late

### What to improve

- add Ollama
- separate chat-provider and embedding-provider capability modeling more clearly
- document provider compatibility and limitations

### Why not now

OpenAI-compatible + Qwen is already enough to validate the current system.

---

## Phase H: Engineering and deployment

**Goal:** make the project easier to maintain, share, and deploy.

**Priority:** later

### What to improve

- Docker
- CI
- baseline tests
- release-ready env docs
- deployment guide

---

## 5. Official execution order

Follow this order unless there is a strong reason to change it:

1. Eval reinforcement
2. Explicit degradation and reliability
3. Retrieval engineering reinforcement
4. Qdrant incremental indexing
5. Agentization upgrades
6. Product experience round two
7. Provider expansion
8. Engineering and deployment

---

## 6. Current project phase snapshot

Approximate current progress:

- Stage 1: minimal runnable agent
  - complete
- Stage 2: tools
  - mostly complete
- Stage 3: RAG
  - mostly complete
- Stage 4: eval
  - first reinforced version complete
- Stage 5: provider abstraction
  - first usable version complete
- Stage 6: engineering
  - first substantial version complete

Approximate overall maturity:

- strong MVP / early product prototype
- not yet a fully hardened agent system

---

## 7. How to use this file during development

When continuing work:

1. identify the next task
2. map it to a phase in this document
3. prefer the highest-priority unfinished phase
4. avoid jumping to later phases unless needed
5. update this roadmap when a phase meaningfully advances

This file should remain the main long-term planning reference for the repository.
