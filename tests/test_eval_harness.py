import json

import pytest

from local_docs_rag_agent.evals.harness import (
    _match_rate,
    _source_rate,
    load_eval_cases,
)
from local_docs_rag_agent.exceptions import DataFormatError


def test_empty_expectations_are_neutral_success() -> None:
    assert _match_rate([], "anything") == 1.0
    assert _source_rate([], []) == 1.0


def test_eval_loader_reports_malformed_line(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "valid"}),
                json.dumps({"question": "invalid", "expected_source_paths": "not-a-list"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataFormatError, match="line 2"):
        load_eval_cases(eval_path)
