import type { CompareResponse } from "../types/api";

type CompareLeaderboardProps = {
  result: CompareResponse | null;
};

export function CompareLeaderboard({ result }: CompareLeaderboardProps) {
  return (
    <article className="panel panel-side">
      <div className="panel-head compact">
        <div>
          <h2>Compare leaderboard</h2>
          <p className="panel-subtle">
            Quick retrieval experiment view for chunk strategies and backends.
          </p>
        </div>
        <span className="meta-pill">{result?.leaderboard.length ?? 0}</span>
      </div>
      {result ? (
        <div className="compare-board">
          {result.leaderboard.length > 0 ? (
            result.leaderboard.map((row, index) => (
              <article key={row.label} className="compare-card">
                <div className="compare-head">
                  <div>
                    <span className="citation-index">#{index + 1}</span>
                    <strong>{row.label}</strong>
                  </div>
                  <span className="meta-pill">{row.avg_response_time_ms.toFixed(2)} ms</span>
                </div>
                <div className="compare-metrics">
                  <span>retrieval span {row.retrieval_span_hit_rate}</span>
                  <span>retrieval source {row.retrieval_source_hit_rate}</span>
                  <span>citation span {row.citation_span_hit_rate}</span>
                  <span>answer hit {row.answer_keyword_hit_rate}</span>
                </div>
              </article>
            ))
          ) : (
            <p className="panel-subtle">No successful compare runs yet.</p>
          )}

          <div className="compare-status-list">
            {result.runs
              .filter((run) => run.status !== "ok")
              .map((run) => (
                <article key={run.label} className="compare-status-card">
                  <strong>{run.label}</strong>
                  <span className="status-chip warn">{run.status}</span>
                  <p>{run.reason ?? run.error ?? "No details"}</p>
                </article>
              ))}
          </div>
        </div>
      ) : (
        <p className="panel-subtle">Run Compare to generate a retrieval leaderboard.</p>
      )}
    </article>
  );
}
