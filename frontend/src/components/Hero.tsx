import { relativeTime } from "../lib/presentation";
import type { AppInfo, Health } from "../types/api";

type HeroProps = {
  health: Health | null;
  info: AppInfo | null;
  documentsCount: number;
  bootstrapError: string | null;
};

export function Hero({ health, info, documentsCount, bootstrapError }: HeroProps) {
  const statusLabel = bootstrapError
    ? "Backend unavailable"
    : health
      ? `Backend ${health.status}`
      : "Checking backend...";

  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">Agent + RAG + Eval</p>
        <h1>Local Docs RAG Agent</h1>
        <p className="lede">
          A local-docs workspace for retrieval, citations, and evaluation with a split frontend and
          backend development flow.
        </p>
      </div>
      <div className="hero-stats">
        <article className="stat-card accent">
          <span className="stat-label">Backend status</span>
          <strong>{statusLabel}</strong>
          <span className="stat-foot">
            {health
              ? `Updated ${relativeTime(health.backend_time_utc)}`
              : "Waiting for health check"}
          </span>
        </article>
        <article className="stat-card">
          <span className="stat-label">Documents</span>
          <strong>{info?.docs_count ?? documentsCount}</strong>
          <span className="stat-foot">{info?.docs_dir ?? "docs/"}</span>
        </article>
        <article className="stat-card">
          <span className="stat-label">Retrieval</span>
          <strong>{info?.vector_backend ?? "loading..."}</strong>
          <span className="stat-foot">top-k {info?.top_k ?? "-"}</span>
        </article>
      </div>
    </section>
  );
}
