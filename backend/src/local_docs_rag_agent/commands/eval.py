"""Runs the eval harness from the command line."""

from __future__ import annotations

import json

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import presenters, rag
from local_docs_rag_agent.evals import harness


def run_eval_command(
    config: app_config.AppConfig, runtime: str | None = None
) -> None:
    """Run the eval harness and print its summary as JSON.

    Args:
      config: The settings to evaluate under.
      runtime: Runtime override for this command, or None to use the
        configured one.
    """
    config = config.with_runtime(runtime)
    rag.ensure_index(config)
    results = harness.run_eval(config)
    summary = presenters.serialize_eval_summary(
        results, runtime=config.agent_runtime, config=config
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
