import type { AppInfo } from "../types/api";

type ConfigPanelProps = {
  info: AppInfo | null;
  error: string | null;
};

export function ConfigPanel({ info, error }: ConfigPanelProps) {
  const fallback = error ? "unavailable" : "loading...";

  return (
    <article className="panel panel-side">
      <h2>Current config</h2>
      {error ? (
        <p className="error-text" role="alert">
          Configuration unavailable: {error}
        </p>
      ) : null}
      <dl className="config-list">
        <div>
          <dt>LLM</dt>
          <dd>{info ? `${info.llm_provider} / ${info.llm_model}` : fallback}</dd>
        </div>
        <div>
          <dt>Embeddings</dt>
          <dd>
            {info ? `${info.embedding_provider} / ${info.embedding_model}` : fallback}
          </dd>
        </div>
        <div>
          <dt>Vector backend</dt>
          <dd>{info?.vector_backend ?? fallback}</dd>
        </div>
        <div>
          <dt>Default runtime</dt>
          <dd>{info?.runtime ?? fallback}</dd>
        </div>
        <div>
          <dt>Retrieval</dt>
          <dd>
            {info
              ? `${info.retrieval_strategy} / top ${info.top_k} / rerank ${info.reranker}`
              : fallback}
          </dd>
        </div>
        <div>
          <dt>Chunking</dt>
          <dd>
            {info
              ? `${info.chunk_strategy} / ${info.chunk_size} size / ${info.chunk_overlap} overlap`
              : fallback}
          </dd>
        </div>
        <div>
          <dt>Exclude patterns</dt>
          <dd>
            {info
              ? info.docs_exclude_patterns.length
                ? info.docs_exclude_patterns.join(", ")
                : "none"
              : fallback}
          </dd>
        </div>
        <div>
          <dt>Qdrant collection</dt>
          <dd>{info?.qdrant_collection ?? fallback}</dd>
        </div>
        <div>
          <dt>Environment proxy</dt>
          <dd>{info ? (info.external_http_trust_env ? "enabled" : "ignored") : fallback}</dd>
        </div>
      </dl>
    </article>
  );
}
