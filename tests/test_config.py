from __future__ import annotations

import pytest

from local_docs_rag_agent import config as config_module
from local_docs_rag_agent.config import AppConfig


@pytest.fixture(autouse=True)
def disable_project_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)


def test_external_http_trust_env_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTERNAL_HTTP_TRUST_ENV", raising=False)

    config = AppConfig.from_env()

    assert config.external_http_trust_env is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_external_http_trust_env_accepts_false_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", value)

    config = AppConfig.from_env()

    assert config.external_http_trust_env is False


def test_external_http_trust_env_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "sometimes")

    with pytest.raises(ValueError, match="EXTERNAL_HTTP_TRUST_ENV"):
        AppConfig.from_env()
