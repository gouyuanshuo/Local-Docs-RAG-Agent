"""The expected-failure taxonomy shared by API, CLI, and eval matrix.

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
        """Record the failure and, where one exists, how to fix it.

        Args:
          message: What went wrong.
          action_hint: What the operator can do about it. Omitted when
            there is no action that reliably helps.
        """
        super().__init__(message)
        self.message = message
        self.action_hint = action_hint

    def __str__(self) -> str:
        """Return the message, with the action hint when there is one."""
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
        """Record a vector-store failure under a stable reason code.

        Args:
          message: What went wrong.
          reason_code: Stable identifier the eval matrix branches on to
            tell 'Qdrant is unreachable here' apart from 'this
            configuration genuinely failed'.
          action_hint: What the operator can do about it.
        """
        super().__init__(message, action_hint=action_hint)
        self.reason_code = reason_code
        self.code = f"vector_store_{reason_code}"
