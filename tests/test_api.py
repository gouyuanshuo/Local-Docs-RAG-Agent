from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_docs_rag_agent.api.app import create_app


def test_info_exposes_external_http_proxy_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "false")
    client = TestClient(create_app())

    response = client.get("/api/info")

    assert response.status_code == 200
    assert response.json()["external_http_trust_env"] is False


def test_expected_application_error_has_stable_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "missing"))
    client = TestClient(create_app())

    response = client.get("/api/documents")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_configuration"
    assert "does not exist" in response.json()["detail"]


def test_ask_rejects_blank_question() -> None:
    response = TestClient(create_app()).post("/api/ask", json={"question": "   "})

    assert response.status_code == 422
