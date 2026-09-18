from __future__ import annotations

import pathlib

import pytest
from fastapi import testclient as fastapi_testclient

from local_docs_rag_agent.api import app as api_app
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.evals import comparison


def _comparison_report() -> dict[str, object]:
    dataset_identity = "d" * 64
    return {
        "num_runs": 6,
        "dataset_identity": dataset_identity,
        "runtimes": ["basic"],
        "chunk_strategies": ["markdown"],
        "vector_backends": ["local", "qdrant"],
        "top_ks": [1, 2, 3],
        "chunk_sizes": [800],
        "chunk_overlaps": [120],
        "retrieval_strategies": ["blended"],
        "rerankers": ["none"],
        "leaderboard": [],
        "unknown_report_field": "ignored by the existing extra policy",
        "runs": [
            {
                "label": "ok-local",
                "status": "ok",
                "dataset_identity": dataset_identity,
                "configuration_identity": "1" * 64,
            },
            {
                "label": "degraded-local",
                "status": "degraded",
                "reason": "embedding provider used fallback mode",
                "dataset_identity": dataset_identity,
                "configuration_identity": "2" * 64,
            },
            {
                "label": "error-local",
                "status": "error",
                "error": "RuntimeError: eval failure",
                "dataset_identity": dataset_identity,
                "configuration_identity": "3" * 64,
            },
            {
                "label": "ok-qdrant",
                "status": "ok",
                "dataset_identity": dataset_identity,
                "configuration_identity": "4" * 64,
                "run_metadata": {
                    "disposable_qdrant_collection": "compare-success",
                },
            },
            {
                "label": "skipped-qdrant",
                "status": "skipped",
                "reason": "qdrant_unreachable",
                "dataset_identity": dataset_identity,
                "configuration_identity": "5" * 64,
            },
            {
                "label": "error-qdrant-cleanup",
                "status": "error",
                "error": "Qdrant comparison cleanup failed",
                "dataset_identity": dataset_identity,
                "configuration_identity": "6" * 64,
                "unknown_run_field": "ignored",
                "run_metadata": {
                    "disposable_qdrant_collection": "compare-orphan",
                    "orphan_recovery_required": True,
                    "pre_cleanup_status": "error",
                    "cleanup_error": "RuntimeError: cleanup failure",
                    "prior_error": "RuntimeError: eval failure",
                    "unknown_metadata_field": "ignored",
                },
            },
        ],
    }


def test_info_exposes_external_http_proxy_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "false")
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.get("/api/info")

    assert response.status_code == 200
    assert response.json()["external_http_trust_env"] is False


def test_expected_application_error_has_stable_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "missing"))
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.get("/api/documents")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_configuration"
    assert "does not exist" in response.json()["detail"]


def test_ask_rejects_blank_question() -> None:
    response = fastapi_testclient.TestClient(api_app.create_app()).post(
        "/api/ask", json={"question": "   "}
    )

    assert response.status_code == 422


def test_ask_rejects_oversized_question() -> None:
    response = fastapi_testclient.TestClient(api_app.create_app()).post(
        "/api/ask", json={"question": "a" * 4001}
    )

    assert response.status_code == 422


def test_eval_compare_schema_preserves_run_provenance() -> None:
    response = schemas.EvalCompareResponse.model_validate(_comparison_report())

    assert response.dataset_identity == "d" * 64
    assert [run.configuration_identity for run in response.runs] == [
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
    ]
    assert {run.status for run in response.runs} == {
        "ok",
        "degraded",
        "skipped",
        "error",
    }
    cleanup_metadata = response.runs[-1].run_metadata
    assert cleanup_metadata is not None
    assert cleanup_metadata.model_dump(exclude_none=True) == {
        "disposable_qdrant_collection": "compare-orphan",
        "orphan_recovery_required": True,
        "pre_cleanup_status": "error",
        "cleanup_error": "RuntimeError: cleanup failure",
        "prior_error": "RuntimeError: eval failure",
    }
    assert "unknown_report_field" not in response.model_dump()
    assert "unknown_run_field" not in response.runs[-1].model_dump()
    assert "unknown_metadata_field" not in cleanup_metadata.model_dump()


def test_eval_compare_endpoint_preserves_run_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def return_report(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return _comparison_report()

    monkeypatch.setattr(comparison, "run_eval_matrix", return_report)
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.post("/api/eval/compare", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_identity"] == "d" * 64
    assert payload["runs"][0]["dataset_identity"] == "d" * 64
    assert payload["runs"][0]["configuration_identity"] == "1" * 64
    assert payload["runs"][-1]["run_metadata"] == {
        "disposable_qdrant_collection": "compare-orphan",
        "orphan_recovery_required": True,
        "pre_cleanup_status": "error",
        "cleanup_error": "RuntimeError: cleanup failure",
        "prior_error": "RuntimeError: eval failure",
    }
