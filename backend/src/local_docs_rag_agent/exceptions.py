"""The expected-failure taxonomy shared by the API, the CLI, and the eval matrix.

Every expected failure carries a stable `code` and an optional `action_hint`
naming the concrete next step. That pairing is what lets one raise site serve
three audiences: the API maps the code to a status, the CLI prints the hint, and
the eval matrix decides whether a cell was skipped or genuinely failed.

Unexpected programming errors deliberately do not derive from `LocalDocsError`;
they stay visible instead of being converted into a tidy failure response.
"""

from __future__ import annotations


class LocalDocsError(Exception):
    """Base class for expected application failures."""

    code = "local_docs_error"

    def __init__(self, message: str, *, action_hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.action_hint = action_hint

    def __str__(self) -> str:
        if self.action_hint is None:
            return self.message
        return f"{self.message}\naction_hint={self.action_hint}"


class ConfigurationError(LocalDocsError, ValueError):
    """Raised when configuration values violate application invariants."""

    code = "invalid_configuration"


class DataFormatError(LocalDocsError):
    """Raised when a persisted project data file is malformed."""

    code = "invalid_data_format"


class ProviderUnavailableError(LocalDocsError):
    """Raised when a live provider is required but unavailable."""

    code = "provider_unavailable"


class VectorStoreError(LocalDocsError):
    """Raised for normalized vector-store failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        action_hint: str | None = None,
    ) -> None:
        super().__init__(message, action_hint=action_hint)
        self.reason_code = reason_code
        self.code = f"vector_store_{reason_code}"
