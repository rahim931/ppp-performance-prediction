class LLMError(Exception):
    """Base error for all LLM client issues."""


class LLMAuthError(LLMError):
    """Authentication or billing failed (e.g., invalid API key, payment required)."""


class LLMRateLimitError(LLMError):
    """Rate limit reached (HTTP 429)."""


class LLMTemporaryError(LLMError):
    """Temporary server/network error worth retrying (e.g., 5xx, timeouts)."""


class LLMBadRequestError(LLMError):
    """Bad request / invalid payload (HTTP 400/422)."""
