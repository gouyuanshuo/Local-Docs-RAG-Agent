type ActionControlsProps = {
  rawPayload: string;
  isActing: boolean;
  onIngest: () => Promise<void>;
  onEval: () => Promise<void>;
  onCompare: () => Promise<void>;
};

export function ActionControls({
  rawPayload,
  isActing,
  onIngest,
  onEval,
  onCompare,
}: ActionControlsProps) {
  return (
    <article className="panel panel-side">
      <h2>Run controls</h2>
      <div className="stack">
        <div className="control-card">
          <h3>Index</h3>
          <p>Refresh the retrieval index from the configured docs directory.</p>
          <button className="action" onClick={() => void onIngest()} disabled={isActing}>
            Run Ingest
          </button>
        </div>
        <div className="control-card">
          <h3>Eval</h3>
          <p>Run the current sample eval set with the active runtime selection.</p>
          <button className="action" onClick={() => void onEval()} disabled={isActing}>
            Run Eval
          </button>
        </div>
        <div className="control-card">
          <h3>Compare</h3>
          <p>Benchmark multiple retrieval setups and surface a simple leaderboard.</p>
          <button className="action" onClick={() => void onCompare()} disabled={isActing}>
            Run Compare
          </button>
        </div>
      </div>
      <details className="debug-panel side-debug">
        <summary>Raw action payload</summary>
        <pre className="output">{rawPayload}</pre>
      </details>
    </article>
  );
}
