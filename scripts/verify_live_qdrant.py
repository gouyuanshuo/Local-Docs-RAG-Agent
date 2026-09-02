"""Strict live gate for the full Qdrant path: ingest, then ask, then eval.

Every stage must complete without degrading. A runtime that fell back, a chat provider
in fallback mode, or embeddings served from the hash fallback all fail the run, because
the point of this check is to prove the live path works end to end.

Run it only against a disposable or explicitly approved collection: unlike
`check_qdrant.py`, this writes.
"""

from __future__ import annotations

import sys

from local_docs_rag_agent.agent import LocalDocsAgent
from local_docs_rag_agent.config import AppConfig
from local_docs_rag_agent.evals.harness import load_eval_cases, run_eval
from local_docs_rag_agent.exceptions import LocalDocsError, ProviderUnavailableError
from local_docs_rag_agent.models import AgentAnswer, AnswerDiagnostics, EvalResult
from local_docs_rag_agent.presenters import serialize_eval_summary
from local_docs_rag_agent.rag.ingest import ingest_documents


def main() -> int:
    stage = "configuration"
    try:
        config = AppConfig.from_env()
        if config.vector_backend != "qdrant":
            raise ProviderUnavailableError("Live verification requires VECTOR_BACKEND=qdrant")

        stage = "ingest"
        changed_chunks = ingest_documents(config)
        print(f"ingest=ok changed_chunks={len(changed_chunks)}")

        cases = load_eval_cases(config.eval_path)
        if not cases:
            raise ProviderUnavailableError("Live verification requires at least one eval case")

        stage = "ask"
        answer = LocalDocsAgent(config).answer(cases[0].question)
        _require_live_answer(answer)
        print(
            "ask=ok "
            f"runtime={answer.diagnostics.actual_runtime} "
            f"chat={answer.diagnostics.chat_provider.mode} "
            f"embedding={answer.diagnostics.embedding_provider.mode} "
            f"citations={len(answer.citations)}"
        )

        stage = "eval"
        results = run_eval(config)
        _require_live_eval(results)
        summary = serialize_eval_summary(
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
    except (LocalDocsError, OSError, ValueError) as exc:
        print(f"stage={stage} status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


def _require_live_answer(answer: AgentAnswer) -> None:
    _require_live_diagnostics(answer.diagnostics, operation="Live ask")


def _require_live_eval(results: list[EvalResult]) -> None:
    for result in results:
        if result.diagnostics is None:
            raise ProviderUnavailableError(
                f"Eval result for {result.question!r} did not include diagnostics"
            )
        _require_live_diagnostics(
            result.diagnostics,
            operation=f"Eval case {result.question!r}",
        )


def _require_live_diagnostics(
    diagnostics: AnswerDiagnostics,
    *,
    operation: str,
) -> None:
    degraded = []
    if diagnostics.actual_runtime != diagnostics.requested_runtime:
        degraded.append(f"runtime={diagnostics.requested_runtime}->{diagnostics.actual_runtime}")
    if diagnostics.chat_provider.mode != "live":
        degraded.append(f"chat={diagnostics.chat_provider.mode}:{diagnostics.chat_provider.reason}")
    if diagnostics.embedding_provider.mode != "live":
        degraded.append(
            "embedding="
            f"{diagnostics.embedding_provider.mode}:{diagnostics.embedding_provider.reason}"
        )
    if degraded:
        raise ProviderUnavailableError(
            f"{operation} degraded instead of completing fully live",
            action_hint="; ".join(degraded),
        )


if __name__ == "__main__":
    raise SystemExit(main())
