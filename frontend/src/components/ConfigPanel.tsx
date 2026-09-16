import type { AppInfo } from "../types/api";

type ConfigPanelProps = {
  info: AppInfo | null;
  bootstrapError: string | null;
};

export function ConfigPanel({ info, bootstrapError }: ConfigPanelProps) {
  return (
    <article className="panel panel-side">
      <h2>Current config</h2>
      {bootstrapError ? <p className="error-text">{bootstrapError}</p> : null}
      <dl className="config-list">
        <div>
          <dt>LLM</dt>
          <dd>{info ? `${info.llm_provider} / ${info.llm_model}` : "loading..."}</dd>
        </div>
        <div>
          <dt>Embeddings</dt>
          <dd>
            {info ? `${info.embedding_provider} / ${info.embedding_model}` : "loading..."}
          </dd>
        </div>
        <div>
          <dt>Vector backend</dt>
          <dd>{info?.vector_backend ?? "loading..."}</dd>
        </div>
        <div>
          <dt>Default runtime</dt>
          <dd>{info?.runtime ?? "loading..."}</dd>
        </div>
        <div>
          <dt>Retrieval</dt>
          <dd>
            {info
              ? `${info.retrieval_strategy} / top ${info.top_k} / rerank ${info.reranker}`
              : "loading..."}
          </dd>
        </div>
        <div>
          <dt>Chunking</dt>
          <dd>
            {info
              ? `${info.chunk_strategy} / ${info.chunk_size} size / ${info.chunk_overlap} overlap`
              : "loading..."}
          </dd>
        </div>
        <div>
          <dt>Exclude patterns</dt>
          <dd>
            {info?.docs_exclude_patterns.length ? info.docs_exclude_patterns.join(", ") : "none"}
          </dd>
        </div>
        <div>
          <dt>Qdrant collection</dt>
          <dd>{info?.qdrant_collection ?? "loading..."}</dd>
        </div>
        <div>
          <dt>Environment proxy</dt>
          <dd>{info ? (info.external_http_trust_env ? "enabled" : "ignored") : "loading..."}</dd>
        </div>
      </dl>
    </article>
  );
}
