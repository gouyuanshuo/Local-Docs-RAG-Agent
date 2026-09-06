"""Runs the eval comparison matrix and writes its report."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping, Sequence

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent.evals import comparison

DEFAULT_COMPARE_OUTPUT_PATH = pathlib.Path("data/evals/compare_latest.json")


def run_eval_compare_command(
    config: app_config.AppConfig,
    requested: Mapping[str, Sequence[object] | None] | None = None,
    output_path: str | None = None,
) -> None:
    """Compare eval results across the requested axes, printing JSON.

    Args:
      config: The baseline settings each cell varies from.
      requested: Values to sweep, keyed by axis name. An omitted axis takes
        its default, so the bare command compares chunk and retrieval
        strategies while holding everything else steady.
      output_path: Where to write the report, or None for the default path.
    """
    payload = comparison.run_eval_matrix(
        config=config,
        requested=requested or {},
        output_path=pathlib.Path(output_path)
        if output_path
        else DEFAULT_COMPARE_OUTPUT_PATH,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))
