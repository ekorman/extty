"""Provider registry and factory functions."""

from extty_infra.providers.base import CloudProvider
from extty_infra.providers.exceptions import ProviderConfigError
from extty_infra.providers.models import ProviderName


def get_provider(name: ProviderName, api_key: str) -> CloudProvider:
    """
    Get a provider instance by name.

    Parameters
    ----------
    name : ProviderName
        The provider name ('lambda', 'vast', or 'prime').
    api_key : str
        The API key for authentication.

    Returns
    -------
    CloudProvider
        An instance of the requested provider.

    Raises
    ------
    ProviderConfigError
        If the provider name is unknown.
    """
    if name == "lambda":
        from extty_infra.providers.lambda_provider import LambdaProvider

        return LambdaProvider(api_key)
    elif name == "vast":
        from extty_infra.providers.vast_provider import VastProvider

        return VastProvider(api_key)
    elif name == "prime":
        from extty_infra.providers.prime_provider import PrimeProvider

        return PrimeProvider(api_key)
    else:
        raise ProviderConfigError(f"Unknown provider: {name}")
