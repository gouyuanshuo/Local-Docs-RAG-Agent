from __future__ import annotations

from fastapi.testclient import TestClient

from local_docs_rag_agent.api.app import create_app


def test_info_exposes_external_http_proxy_setting(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "false")
    client = TestClient(create_app())

    response = client.get("/api/info")

    assert response.status_code == 200
    assert response.json()["external_http_trust_env"] is False
