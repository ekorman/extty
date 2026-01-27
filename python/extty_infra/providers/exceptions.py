"""Exceptions for cloud providers."""


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ProviderAPIError(ProviderError):
    """Exception raised for provider API errors."""

    def __init__(self, code: str, message: str, suggestion: str | None = None):
        self.code = code
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{code}: {message}")


class ProviderConfigError(ProviderError):
    """Exception raised for provider configuration errors."""

    pass
