import { ActionControls } from "./components/ActionControls";
import { AskPanel } from "./components/AskPanel";
import { CompareLeaderboard } from "./components/CompareLeaderboard";
import { ConfigPanel } from "./components/ConfigPanel";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { Hero } from "./components/Hero";
import { useRagWorkspace } from "./hooks/useRagWorkspace";

export default function App() {
  const workspace = useRagWorkspace();

  return (
    <main className="shell">
      <Hero
        health={workspace.health}
        info={workspace.info}
        documentsCount={workspace.documents.length}
        bootstrapError={workspace.bootstrapError}
      />

      <section className="workspace">
        <div className="main-column">
          <AskPanel
            runtime={workspace.runtime}
            question={workspace.question}
            result={workspace.askResult}
            rawPayload={workspace.askRaw}
            isAsking={workspace.isAsking}
            onRuntimeChange={workspace.setRuntime}
            onQuestionChange={workspace.setQuestion}
            onAsk={workspace.ask}
          />
        </div>

        <aside className="side-column">
          <ActionControls
            rawPayload={workspace.actionRaw}
            isActing={workspace.isActing}
            onIngest={workspace.ingest}
            onEval={workspace.evaluate}
            onCompare={workspace.compare}
          />
          <CompareLeaderboard result={workspace.compareResult} />
          <ConfigPanel info={workspace.info} bootstrapError={workspace.bootstrapError} />
          <DocumentsPanel documents={workspace.documents} />
        </aside>
      </section>
    </main>
  );
}
