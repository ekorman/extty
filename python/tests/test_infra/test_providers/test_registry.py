"""Tests for provider registry."""

import pytest
from unittest.mock import patch

from extty_infra.providers.registry import get_provider
from extty_infra.providers.lambda_provider import LambdaProvider
from extty_infra.providers.vast_provider import VastProvider
from extty_infra.providers.prime_provider import PrimeProvider
from extty_infra.providers.exceptions import ProviderConfigError


def test_get_lambda_provider():
    """Test getting Lambda provider."""
    with patch("httpx.Client"):
        provider = get_provider("lambda", "test-key")
        assert isinstance(provider, LambdaProvider)
        assert provider.name == "lambda"


def test_get_vast_provider():
    """Test getting Vast provider."""
    with patch("httpx.Client"):
        provider = get_provider("vast", "test-key")
        assert isinstance(provider, VastProvider)
        assert provider.name == "vast"


def test_get_prime_provider():
    """Test getting Prime provider."""
    with patch("httpx.Client"):
        provider = get_provider("prime", "test-key")
        assert isinstance(provider, PrimeProvider)
        assert provider.name == "prime"


def test_get_unknown_provider():
    """Test getting unknown provider raises error."""
    with pytest.raises(ProviderConfigError) as exc:
        get_provider("unknown", "test-key")  # type: ignore
    assert "Unknown provider" in str(exc.value)
