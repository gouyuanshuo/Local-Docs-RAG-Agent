"""Classifies provider failures as transient or permanent.

Transient failures are worth retrying; permanent ones are not, and retrying them
only delays the report. Classification looks at the exception type, an HTTP
status code when one is attached, and finally the message text, because
OpenAI-compatible endpoints vary in which of the three they populate.
"""

from __future__ import annotations

TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
TRANSIENT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
)
TRANSIENT_MESSAGE_TOKENS = (
    "connection aborted",
    "connection reset",
    "temporarily",
    "timed out",
    "timeout",
    "try again",
)


def is_transient_provider_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in TRANSIENT_ERROR_NAMES:
        return True
    if getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    message = str(exc).lower()
    return any(token in message for token in TRANSIENT_MESSAGE_TOKENS)


def provider_error_reason(exc: Exception, *, limit: int = 240) -> str:
    class_name = exc.__class__.__name__
    message = " ".join(str(exc).split())
    if len(message) > limit:
        message = f"{message[:limit].rstrip()}..."
    if not message:
        return f"provider_error:{class_name}"
    return f"provider_error:{class_name}:{message}"
