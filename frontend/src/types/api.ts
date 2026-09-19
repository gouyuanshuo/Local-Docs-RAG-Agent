export type RuntimeName = "basic" | "agents_sdk";
export type RuntimeSelection = "" | RuntimeName;
export type VectorBackendName = "local" | "qdrant";
export type ChunkStrategyName = "fixed" | "paragraph" | "markdown";
export type RetrievalStrategyName = "blended" | "dense" | "lexical" | "hybrid_rrf";
export type RerankerName = "none" | "llm";

export type ProviderMode = "ready" | "live" | "fallback" | "unknown";

export type ProviderStatus = {
  provider: string;
  mode: ProviderMode;
  reason: string | null;
};

export type CitationSpan = {
  source_path: string;
  chunk_id: string;
  chunk_index: number;
  start_char: number;
  end_char: number;
  text: string;
  source_id: string | null;
};

export type AnswerDiagnostics = {
  requested_runtime: RuntimeName;
  actual_runtime: RuntimeName;
  vector_backend: VectorBackendName;
  chat_provider: ProviderStatus;
  embedding_provider: ProviderStatus;
  reranker: ProviderStatus;
};

export type AskResponse = {
  question: string;
  answer: string;
  citations: string[];
  citation_spans: CitationSpan[];
  runtime: RuntimeName;
  diagnostics: AnswerDiagnostics;
};

export type AppInfo = {
  name: string;
  runtime: RuntimeName;
  vector_backend: VectorBackendName;
  docs_dir: string;
  docs_exclude_patterns: string[];
  docs_count: number;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  top_k: number;
  retrieval_strategy: RetrievalStrategyName;
  reranker: RerankerName;
  chunk_strategy: ChunkStrategyName;
  chunk_size: number;
  chunk_overlap: number;
  qdrant_collection: string;
  external_http_trust_env: boolean;
};

export type Health = {
  status: "ok";
  backend_time_utc: string;
};

export type DocumentsResponse = {
  count: number;
  documents: string[];
};

export type RetrievalConfig = {
  vector_backend: VectorBackendName;
  chunk_strategy: ChunkStrategyName;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  retrieval_strategy: RetrievalStrategyName;
  retrieval_candidate_k: number;
  rrf_k: number;
  reranker: RerankerName;
  rerank_candidate_k: number;
  docs_dir: string;
  docs_exclude_patterns: string[];
};

export type EvalResult = {
  question: string;
  answer: string;
  citations: string[];
  retrieved_sources: string[];
  answer_keyword_hit_rate: number;
  retrieval_source_hit_rate: number;
  retrieval_span_hit_rate: number;
  retrieval_reciprocal_rank: number;
  retrieval_precision: number;
  citation_source_hit_rate: number;
  citation_span_hit_rate: number;
  response_time_ms: number;
  diagnostics: AnswerDiagnostics | null;
  expected_source_paths: string[];
  expected_answer_keywords: string[];
  expected_span_keywords: string[];
  expected_retrieval_keywords: string[];
  failure_reasons: string[];
};

export type EvalSummary = {
  num_cases: number;
  runtime: RuntimeName;
  retrieval_config: RetrievalConfig | null;
  answer_keyword_hit_rate: number;
  retrieval_source_hit_rate: number;
  retrieval_span_hit_rate: number;
  retrieval_reciprocal_rank: number;
  retrieval_precision: number;
  citation_source_hit_rate: number;
  citation_span_hit_rate: number;
  avg_response_time_ms: number;
  avg_keyword_hit_rate: number;
  source_hit_rate: number;
  avg_citation_span_hit_rate: number;
  results: EvalResult[];
};

export type CompareLeaderboardRow = {
  label: string;
  answer_keyword_hit_rate: number;
  retrieval_source_hit_rate: number;
  retrieval_span_hit_rate: number;
  // The rank-aware pair the board is ordered by. `retrieval_span_hit_rate`
  // cannot tell first place from fourth, and rises with a wider top_k.
  retrieval_reciprocal_rank: number;
  retrieval_precision: number;
  citation_span_hit_rate: number;
  avg_response_time_ms: number;
};

export type CompareRunStatus = "ok" | "skipped" | "error" | "degraded";

export type CompareRunMetadata = {
  disposable_qdrant_collection: string;
  orphan_recovery_required: boolean;
  pre_cleanup_status: CompareRunStatus | null;
  cleanup_error: string | null;
  prior_error: string | null;
};

export type CompareRun = {
  label: string;
  status: CompareRunStatus;
  dataset_identity: string;
  configuration_identity: string;
  run_metadata: CompareRunMetadata | null;
  reason: string | null;
  error: string | null;
  retrieval_config: RetrievalConfig | null;
  runtime: RuntimeName | null;
  summary: EvalSummary | null;
};

export type CompareResponse = {
  num_runs: number;
  dataset_identity: string;
  runtimes: RuntimeName[];
  chunk_strategies: ChunkStrategyName[];
  vector_backends: VectorBackendName[];
  top_ks: number[];
  chunk_sizes: number[];
  chunk_overlaps: number[];
  retrieval_strategies: RetrievalStrategyName[];
  rerankers: RerankerName[];
  leaderboard: CompareLeaderboardRow[];
  runs: CompareRun[];
};

export type ApiErrorPayload = {
  code?: string;
  detail?: unknown;
  action_hint?: string | null;
};
