"""Typed environment readers and value validators used to build :class:`AppConfig`.

``config.py`` declares *what* the application is configured with; this module owns
*how* each value is read from the process environment and checked. Splitting the two
keeps the configuration schema readable as a list of fields, and guarantees that every
malformed value raises the same :class:`ConfigurationError` shape no matter which
setting produced it.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import TypeVar, cast

from dotenv import load_dotenv

from local_docs_rag_agent.exceptions import ConfigurationError

ChoiceT = TypeVar("ChoiceT", bound=str)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def load_project_dotenv() -> None:
    """Load a project-local ``.env`` without overriding already-exported variables."""

    load_dotenv(override=False)


def env_text(name: str, default: str) -> str:
    """Return a raw string setting, falling back to ``default`` when unset or blank."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


def env_optional_text(name: str) -> str | None:
    """Return a raw string setting, or ``None`` when the variable is unset or blank."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value


def env_first_text(names: tuple[str, ...], default: str) -> str:
    """Return the first non-blank setting among ``names``, else ``default``.

    Used for settings that accept a provider-specific alias alongside the canonical
    variable, such as ``LLM_MODEL`` falling back to ``OPENAI_MODEL``.
    """

    for name in names:
        value = env_optional_text(name)
        if value is not None:
            return value
    return default


def env_first_optional(names: tuple[str, ...]) -> str | None:
    """Return the first non-blank setting among ``names``, or ``None`` if all are unset."""

    for name in names:
        value = env_optional_text(name)
        if value is not None:
            return value
    return None


def env_token(name: str, default: str) -> str:
    """Return a stripped, lowercased setting for case-insensitive option names."""

    return env_text(name, default).strip().lower()


def env_choice(name: str, default: ChoiceT, choices: tuple[ChoiceT, ...]) -> ChoiceT:
    """Return a setting constrained to ``choices``, preserving its narrow literal type."""

    value = env_token(name, default)
    require_choice(name, value, choices)
    return cast(ChoiceT, value)


def env_int(name: str, default: int) -> int:
    """Return an integer setting, falling back to ``default`` when unset or blank."""

    value = os.getenv(name)
    if not value:
        return default
    return _parse_int(name, value)


def env_optional_int(name: str) -> int | None:
    """Return an integer setting, or ``None`` when the variable is unset or blank."""

    value = os.getenv(name)
    if not value:
        return None
    return _parse_int(name, value)


def env_bool(name: str, default: bool) -> bool:
    """Return a boolean setting parsed from the usual truthy/falsy spellings."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    allowed = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ConfigurationError(f"{name} must be one of: {allowed}")


def env_list(name: str) -> list[str]:
    """Return a comma-separated setting as a list, dropping blank entries."""

    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def env_path(name: str, default: str) -> Path:
    """Return a filesystem-path setting as a :class:`~pathlib.Path`."""

    return Path(env_text(name, default))


def require_choice(name: str, value: str, choices: Collection[str]) -> None:
    """Raise unless ``value`` is one of the supported ``choices``."""

    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigurationError(f"{name} must be one of: {allowed}; got {value!r}")


def require_positive(name: str, value: int) -> None:
    """Raise unless ``value`` is greater than zero."""

    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0, got {value}")


def require_non_negative(name: str, value: int) -> None:
    """Raise unless ``value`` is zero or greater."""

    if value < 0:
        raise ConfigurationError(f"{name} must be at least 0, got {value}")


def require_non_empty(name: str, value: str) -> None:
    """Raise unless ``value`` contains at least one non-whitespace character."""

    if not value.strip():
        raise ConfigurationError(f"{name} must not be empty")


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc
