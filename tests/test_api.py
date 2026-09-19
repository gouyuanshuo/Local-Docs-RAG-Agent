from __future__ import annotations

import json
import pathlib

import httpx2
import pytest
import qdrant_client
from fastapi import testclient as fastapi_testclient

from local_docs_rag_agent import agent, presenters, rag
from local_docs_rag_agent.api import app as api_app
from local_docs_rag_agent.api import schemas
from local_docs_rag_agent.core import models
from local_docs_rag_agent.evals import comparison, harness
from local_docs_rag_agent.providers import factory as provider_factory


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


def _configure_qdrant_http_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    *,
    qdrant_url: str,
    qdrant_api_key: str,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "source.md").write_text(
        "# Safe test source\n\nHTTP boundary evidence.\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "question": "Where is the HTTP boundary evidence?",
                "expected_answer_keywords": ["HTTP boundary evidence"],
                "expected_source_paths": ["source.md"],
                "expected_span_keywords": ["HTTP boundary evidence"],
                "expected_retrieval_keywords": ["HTTP boundary evidence"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    environment = {
        "DOCS_DIR": str(docs_dir),
        "EVAL_PATH": str(eval_path),
        "INDEX_PATH": str(tmp_path / "index" / "chunks.jsonl"),
        "INGEST_MANIFEST_PATH": str(tmp_path / "index" / "manifest.json"),
        "VECTOR_BACKEND": "qdrant",
        "QDRANT_URL": qdrant_url,
        "QDRANT_API_KEY": qdrant_api_key,
        "QDRANT_COLLECTION": "safe-http-collection",
        "EXTERNAL_HTTP_TRUST_ENV": "false",
        "AGENT_RUNTIME": "basic",
        "CHUNK_STRATEGY": "markdown",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "120",
        "TOP_K": "4",
        "RETRIEVAL_STRATEGY": "blended",
        "RERANKER": "none",
        "LLM_API_KEY": "",
        "OPENAI_API_KEY": "",
        "EMBEDDING_API_KEY": "",
        "LLM_BASE_URL": "",
        "EMBEDDING_BASE_URL": "",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def _compare_request(vector_backends: list[str]) -> dict[str, object]:
    return {
        "runtimes": ["basic"],
        "chunk_strategies": ["markdown"],
        "vector_backends": vector_backends,
        "top_ks": [4],
        "chunk_sizes": [800],
        "chunk_overlaps": [120],
        "retrieval_strategies": ["blended"],
        "rerankers": ["none"],
    }


def _response_surface(response: httpx2.Response) -> str:
    return "\n".join(
        (
            str(response.status_code),
            response.text,
            json.dumps(dict(response.headers), sort_keys=True),
        )
    )


def _assert_response_excludes(
    response: httpx2.Response, sentinels: tuple[str, ...]
) -> None:
    surface = _response_surface(response)
    for sentinel in sentinels:
        assert sentinel not in surface


def _install_echoing_qdrant_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configured_url: str,
    api_key: str,
    raw_marker: str,
) -> None:
    def fail_client(**kwargs: object) -> None:
        raise RuntimeError(
            f"{raw_marker}; url={configured_url}; key={api_key}; "
            f"kwargs={kwargs!r}"
        )

    monkeypatch.setattr(qdrant_client, "QdrantClient", fail_client)


def test_create_app_without_discovered_frontend_serves_api_and_json_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    isolated_module = (
        tmp_path / "site-packages" / "local_docs_rag_agent" / "api" / "app.py"
    )
    monkeypatch.setattr(api_app, "__file__", str(isolated_module))

    client = fastapi_testclient.TestClient(api_app.create_app())

    health_response = client.get("/api/health")
    root_response = client.get("/")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert root_response.status_code == 200
    assert root_response.json() == {
        "name": "Local Docs RAG Agent API",
        "message": (
            "Frontend dev server not running. Start it with `pnpm run dev`."
        ),
    }


def test_create_app_serves_explicit_frontend_build(
    tmp_path: pathlib.Path,
) -> None:
    frontend_dist = tmp_path / "explicit-frontend"
    assets_dir = frontend_dist / "assets"
    assets_dir.mkdir(parents=True)
    (frontend_dist / "index.html").write_text(
        "<main>explicit frontend</main>",
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text(
        'document.body.dataset.bundle = "explicit";',
        encoding="utf-8",
    )

    client = fastapi_testclient.TestClient(
        api_app.create_app(frontend_dist_dir=frontend_dist)
    )

    index_response = client.get("/")
    asset_response = client.get("/assets/app.js")

    assert index_response.status_code == 200
    assert index_response.text == "<main>explicit frontend</main>"
    assert asset_response.status_code == 200
    assert asset_response.text == ('document.body.dataset.bundle = "explicit";')


def test_create_app_rejects_invalid_explicit_frontend_without_fallback(
    tmp_path: pathlib.Path,
) -> None:
    invalid_dist = tmp_path / "private-invalid-frontend-sentinel"
    invalid_dist.mkdir()
    (invalid_dist / "index.html").write_text(
        "<main>incomplete build</main>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        api_app.create_app(frontend_dist_dir=invalid_dist)

    assert str(exc_info.value) == (
        "frontend_dist_dir must contain index.html and an assets directory"
    )
    assert "private-invalid-frontend-sentinel" not in str(exc_info.value)


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


def test_ask_exposes_answer_local_citation_source_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.setenv("AGENT_RUNTIME", "basic")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")

    def skip_index(config: object) -> None:
        del config

    def answer_with_source_id(
        self: agent.LocalDocsAgent,
        question: str,
    ) -> models.AgentAnswer:
        del self
        live_status = models.ProviderStatus(provider="test", mode="live")
        return models.AgentAnswer(
            question=question,
            answer="The answer is grounded in the cited span [S1].",
            citations=["docs/source.md"],
            citation_spans=[
                models.CitationSpan(
                    source_path="docs/source.md",
                    chunk_id="chunk-1",
                    chunk_index=0,
                    start_char=10,
                    end_char=34,
                    text="grounded citation text",
                    source_id="S1",
                )
            ],
            retrieved_chunks=[],
            diagnostics=models.AnswerDiagnostics(
                requested_runtime="basic",
                actual_runtime="basic",
                vector_backend="local",
                chat_provider=live_status,
                embedding_provider=live_status,
                reranker=models.ProviderStatus(
                    provider="none",
                    mode="ready",
                    reason="reranker_disabled",
                ),
            ),
        )

    monkeypatch.setattr(rag, "ensure_index", skip_index)
    monkeypatch.setattr(
        agent.LocalDocsAgent,
        "answer",
        answer_with_source_id,
    )
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.post(
        "/api/ask",
        json={"question": "Which source supports the answer?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].endswith("[S1].")
    assert payload["citation_spans"] == [
        {
            "source_path": "docs/source.md",
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "start_char": 10,
            "end_char": 34,
            "text": "grounded citation text",
            "source_id": "S1",
        }
    ]


def test_citation_span_schema_serializes_missing_source_id_as_null() -> None:
    span = models.CitationSpan(
        source_path="legacy.md",
        chunk_id="legacy-chunk",
        chunk_index=0,
        start_char=0,
        end_char=11,
        text="legacy span",
    )
    expected = {
        "source_path": "legacy.md",
        "chunk_id": "legacy-chunk",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 11,
        "text": "legacy span",
        "source_id": None,
    }
    payload = presenters.serialize_citation_span(span)

    assert payload == expected

    response = schemas.CitationSpanResponse.model_validate(payload)

    assert response.model_dump(mode="json") == expected


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


def test_eval_compare_rejects_malformed_qdrant_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    sentinel_user = "invalid-user-sentinel"
    sentinel_password = "invalid-password-sentinel"
    sentinel_path = "invalid-path-sentinel"
    sentinel_query = "invalid-query-sentinel"
    api_key = "invalid-api-key-sentinel"
    configured_url = (
        f"https://{sentinel_user}:{sentinel_password}@"
        f"bad\N{IDEOGRAPHIC FULL STOP}host.invalid/"
        f"{sentinel_path}?key={sentinel_query}"
    )
    _configure_qdrant_http_environment(
        monkeypatch,
        tmp_path,
        qdrant_url=configured_url,
        qdrant_api_key=api_key,
    )
    side_effects: list[str] = []

    def unexpected_side_effect(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("comparison side effect preceded target preflight")

    monkeypatch.setattr(harness, "load_eval_cases", unexpected_side_effect)
    monkeypatch.setattr(rag, "read_source_texts", unexpected_side_effect)
    monkeypatch.setattr(rag, "ingest_documents", unexpected_side_effect)
    monkeypatch.setattr(
        rag, "initialize_owned_qdrant_index", unexpected_side_effect
    )
    monkeypatch.setattr(harness, "run_eval", unexpected_side_effect)
    monkeypatch.setattr(
        provider_factory,
        "build_embedding_provider",
        unexpected_side_effect,
    )
    monkeypatch.setattr(qdrant_client, "QdrantClient", unexpected_side_effect)
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.post(
        "/api/eval/compare",
        json=_compare_request(["local", "qdrant"]),
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_configuration",
        "detail": "QDRANT_URL must be a valid HTTP(S) endpoint",
        "action_hint": None,
    }
    assert side_effects == []
    _assert_response_excludes(
        response,
        (
            sentinel_user,
            sentinel_password,
            sentinel_path,
            sentinel_query,
            api_key,
            configured_url,
        ),
    )


def test_eval_compare_redacts_qdrant_client_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    configured_url = (
        "https://client-user-sentinel:client-password-sentinel@"
        "b\N{LATIN SMALL LETTER U WITH DIAERESIS}cher.example/"
        "client-path-sentinel?key=client-query-sentinel"
        "#client-fragment-sentinel"
    )
    api_key = "client-api-key-sentinel"
    raw_marker = "client-raw-message-sentinel"
    _configure_qdrant_http_environment(
        monkeypatch,
        tmp_path,
        qdrant_url=configured_url,
        qdrant_api_key=api_key,
    )
    _install_echoing_qdrant_failure(
        monkeypatch,
        configured_url=configured_url,
        api_key=api_key,
        raw_marker=raw_marker,
    )
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.post(
        "/api/eval/compare",
        json=_compare_request(["qdrant"]),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["num_runs"] == 1
    run = payload["runs"][0]
    assert run["status"] == "error"
    assert run["dataset_identity"] == payload["dataset_identity"]
    assert isinstance(run["configuration_identity"], str)
    assert len(run["configuration_identity"]) == 64
    metadata = run["run_metadata"]
    collection_name = metadata["disposable_qdrant_collection"]
    assert collection_name.startswith("safe-http-collection-eval-")
    assert metadata["orphan_recovery_required"] is False
    error = run["error"]
    assert "VectorStoreError: Qdrant operation failed." in error
    assert "operation=client_init" in error
    assert f"collection={collection_name}" in error
    assert "error_type=RuntimeError" in error
    _assert_response_excludes(
        response,
        (
            "client-user-sentinel",
            "client-password-sentinel",
            "client-path-sentinel",
            "client-query-sentinel",
            "client-fragment-sentinel",
            api_key,
            raw_marker,
            configured_url,
            "kwargs",
        ),
    )


def test_ingest_redacts_qdrant_client_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    configured_url = (
        "https://ingest-user-sentinel:ingest-password-sentinel@"
        "qdrant.example/ingest-path-sentinel?key=ingest-query-sentinel"
        "#ingest-fragment-sentinel"
    )
    api_key = "ingest-api-key-sentinel"
    raw_marker = "ingest-raw-message-sentinel"
    _configure_qdrant_http_environment(
        monkeypatch,
        tmp_path,
        qdrant_url=configured_url,
        qdrant_api_key=api_key,
    )
    _install_echoing_qdrant_failure(
        monkeypatch,
        configured_url=configured_url,
        api_key=api_key,
        raw_marker=raw_marker,
    )
    client = fastapi_testclient.TestClient(api_app.create_app())

    response = client.post("/api/ingest")

    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == "vector_store_operation_failed"
    assert payload["action_hint"] is None
    detail = payload["detail"]
    assert "Qdrant operation failed." in detail
    assert "operation=client_init" in detail
    assert "collection=safe-http-collection" in detail
    assert "error_type=RuntimeError" in detail
    _assert_response_excludes(
        response,
        (
            "ingest-user-sentinel",
            "ingest-password-sentinel",
            "ingest-path-sentinel",
            "ingest-query-sentinel",
            "ingest-fragment-sentinel",
            api_key,
            raw_marker,
            configured_url,
            "kwargs",
        ),
    )
