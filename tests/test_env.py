from __future__ import annotations

import pytest

from local_docs_rag_agent import constants, env, exceptions


def test_blank_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_SETTING", "   ")

    assert env.optional_text("EXAMPLE_SETTING") is None
    assert env.token("EXAMPLE_SETTING", "fallback") == "fallback"
    assert env.path("EXAMPLE_SETTING", "docs").name == "docs"


def test_token_is_normalized_for_case_insensitive_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_SETTING", "  Chat_Completions  ")

    assert env.token("EXAMPLE_SETTING", "responses") == "chat_completions"


def test_first_non_blank_alias_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIMARY_SETTING", "")
    monkeypatch.setenv("FALLBACK_SETTING", "from-fallback")

    names = ("PRIMARY_SETTING", "FALLBACK_SETTING")
    assert env.first_text(names, "default") == "from-fallback"
    assert env.first_optional(names) == "from-fallback"


def test_first_optional_is_none_when_every_alias_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRIMARY_SETTING", raising=False)
    monkeypatch.delenv("FALLBACK_SETTING", raising=False)

    assert env.first_optional(("PRIMARY_SETTING", "FALLBACK_SETTING")) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_bool_accepts_truthy_spellings(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EXAMPLE_FLAG", value)

    assert env.boolean("EXAMPLE_FLAG", False) is True


def test_bool_rejects_unrecognized_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_FLAG", "maybe")

    with pytest.raises(exceptions.ConfigurationError, match="EXAMPLE_FLAG"):
        env.boolean("EXAMPLE_FLAG", False)


def test_int_reports_the_variable_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_COUNT", "not-a-number")

    with pytest.raises(
        exceptions.ConfigurationError, match="EXAMPLE_COUNT must be an integer"
    ):
        env.integer("EXAMPLE_COUNT", 1)

    with pytest.raises(
        exceptions.ConfigurationError, match="EXAMPLE_COUNT must be an integer"
    ):
        env.optional_integer("EXAMPLE_COUNT")


def test_list_drops_blank_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_PATTERNS", " a , ,b,, c ")

    assert env.string_list("EXAMPLE_PATTERNS") == ["a", "b", "c"]


def test_choice_is_constrained_to_the_shared_option_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_RUNTIME", "agents_sdk")
    assert (
        env.choice(
            "AGENT_RUNTIME",
            constants.DEFAULT_AGENT_RUNTIME,
            constants.AGENT_RUNTIMES,
        )
        == "agents_sdk"
    )

    monkeypatch.setenv("AGENT_RUNTIME", "telepathy")
    with pytest.raises(
        exceptions.ConfigurationError, match="AGENT_RUNTIME must be one of"
    ):
        env.choice(
            "AGENT_RUNTIME",
            constants.DEFAULT_AGENT_RUNTIME,
            constants.AGENT_RUNTIMES,
        )


def test_require_non_empty_rejects_whitespace() -> None:
    with pytest.raises(
        exceptions.ConfigurationError, match="EXAMPLE_NAME must not be empty"
    ):
        env.require_non_empty("EXAMPLE_NAME", "   ")
