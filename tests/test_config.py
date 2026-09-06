from __future__ import annotations

import pytest

from local_docs_rag_agent import config as app_config
from local_docs_rag_agent import exceptions


def test_external_http_trust_env_defaults_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXTERNAL_HTTP_TRUST_ENV", raising=False)

    config = app_config.AppConfig.from_env()

    assert config.external_http_trust_env is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_external_http_trust_env_accepts_false_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", value)

    config = app_config.AppConfig.from_env()

    assert config.external_http_trust_env is False


def test_external_http_trust_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTERNAL_HTTP_TRUST_ENV", "sometimes")

    with pytest.raises(ValueError, match="EXTERNAL_HTTP_TRUST_ENV"):
        app_config.AppConfig.from_env()


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(exceptions.ConfigurationError, match="CHUNK_OVERLAP"):
        app_config.AppConfig.from_env().with_overrides(
            chunk_size=10, chunk_overlap=10
        )


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(
        exceptions.ConfigurationError, match="Unknown AppConfig"
    ):
        app_config.AppConfig.from_env().with_overrides(does_not_exist=True)
