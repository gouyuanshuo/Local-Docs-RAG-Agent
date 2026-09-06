from __future__ import annotations

import pathlib

import pytest
from fastapi import testclient as fastapi_testclient

from local_docs_rag_agent.api import app as api_app


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
