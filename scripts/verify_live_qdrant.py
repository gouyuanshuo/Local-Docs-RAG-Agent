"""Strict live gate for the full Qdrant path: ingest, then ask, then eval.

Every stage must complete without degrading. A runtime that fell back, a chat
provider in fallback mode, or embeddings served from the hash fallback all fail
the run, because the point of this check is to prove the live path works end to
end.

Run it only against a disposable or explicitly approved collection: unlike
`check_qdrant.py`, this writes.
"""

from __future__ import annotations

import sys

from local_docs_rag_agent import agent, exceptions, models, presenters, rag
from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.evals import harness


def main() -> int:
    """Verify a full live ingest, ask, and eval against Qdrant.

    Returns:
      0 when the check passes, 1 when it does not. The failure is
      printed rather than raised, so this reads as a check result
      rather than a crash.
    """
    stage = "configuration"
    try:
        config = app_config.AppConfig.from_env()
        if config.vector_backend != "qdrant":
            raise exceptions.ProviderUnavailableError(
                "Live verification requires VECTOR_BACKEND=qdrant"
            )

        stage = "ingest"
        changed_chunks = rag.ingest_documents(config)
        print(f"ingest=ok changed_chunks={len(changed_chunks)}")

        cases = harness.load_eval_cases(config.eval_path)
        if not cases:
            raise exceptions.ProviderUnavailableError(
                "Live verification requires at least one eval case"
            )

        stage = "ask"
        answer = agent.LocalDocsAgent(config).answer(cases[0].question)
        _require_live_answer(answer)
        print(
            "ask=ok "
            f"runtime={answer.diagnostics.actual_runtime} "
            f"chat={answer.diagnostics.chat_provider.mode} "
            f"embedding={answer.diagnostics.embedding_provider.mode} "
            f"citations={len(answer.citations)}"
        )

        stage = "eval"
        results = harness.run_eval(config)
        _require_live_eval(results)
        summary = presenters.serialize_eval_summary(
            results,
            runtime=config.agent_runtime,
            config=config,
        )
        print(
            "eval=ok "
            f"cases={summary['num_cases']} "
            f"retrieval_source_hit_rate={summary['retrieval_source_hit_rate']} "
            f"answer_keyword_hit_rate={summary['answer_keyword_hit_rate']}"
        )
    except (exceptions.LocalDocsError, OSError, ValueError) as exc:
        print(
            f"stage={stage} status=failed error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


def _require_live_answer(answer: models.AgentAnswer) -> None:
    _require_live_diagnostics(answer.diagnostics, operation="Live ask")


def _require_live_eval(results: list[models.EvalResult]) -> None:
    for result in results:
        if result.diagnostics is None:
            raise exceptions.ProviderUnavailableError(
                f"Eval result for {result.question!r} did not include "
                "diagnostics"
            )
        _require_live_diagnostics(
            result.diagnostics,
            operation=f"Eval case {result.question!r}",
        )


def _require_live_diagnostics(
    diagnostics: models.AnswerDiagnostics,
    *,
    operation: str,
) -> None:
    degraded = []
    if diagnostics.actual_runtime != diagnostics.requested_runtime:
        degraded.append(
            f"runtime={diagnostics.requested_runtime}"
            f"->{diagnostics.actual_runtime}"
        )
    if diagnostics.chat_provider.mode != "live":
        degraded.append(
            f"chat={diagnostics.chat_provider.mode}"
            f":{diagnostics.chat_provider.reason}"
        )
    if diagnostics.embedding_provider.mode != "live":
        degraded.append(
            "embedding="
            f"{diagnostics.embedding_provider.mode}"
            f":{diagnostics.embedding_provider.reason}"
        )
    if degraded:
        raise exceptions.ProviderUnavailableError(
            f"{operation} degraded instead of completing fully live",
            action_hint="; ".join(degraded),
        )


if __name__ == "__main__":
    raise SystemExit(main())
