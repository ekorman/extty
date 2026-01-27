"""Configuration file handling for ~/.ex/infra/config.toml."""

import tomllib
from pathlib import Path

import tomli_w

from extty_infra.models import InfraConfig
from extty_infra.providers.models import ProviderName

CONFIG_DIR = Path.home() / ".ex" / "infra"
CONFIG_FILE = CONFIG_DIR / "config.toml"

PROVIDER_KEYS = {"api_key", "default_region", "default_instance_type", "ssh_key_name"}
TOP_LEVEL_KEYS = {"default_provider"} | PROVIDER_KEYS
VALID_PROVIDERS: set[ProviderName] = {"lambda", "vast", "prime"}


def ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> InfraConfig:
    """Load configuration from TOML file."""
    if not CONFIG_FILE.exists():
        return InfraConfig()

    with open(CONFIG_FILE, "rb") as f:
        data = tomllib.load(f)
    return InfraConfig.model_validate(data)


def load_raw_config() -> dict:
    """Load raw configuration dict from TOML file."""
    if not CONFIG_FILE.exists():
        return {}

    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def save_raw_config(data: dict) -> None:
    """Save raw configuration dict to TOML file."""
    ensure_config_dir()
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(data, f)


def save_config(config: InfraConfig) -> None:
    """Save configuration to TOML file."""
    ensure_config_dir()

    data: dict = {}

    if config.default_provider != "lambda":
        data["default_provider"] = config.default_provider

    for key in ["api_key", "default_region", "default_instance_type", "ssh_key_name"]:
        value = getattr(config, key)
        if value is not None:
            data[key] = value

    for provider in ["lambda", "vast", "prime"]:
        if provider == "lambda":
            provider_config = config.lambda_config
        else:
            provider_config = getattr(config, provider)

        provider_data = {}
        for key in [
            "api_key",
            "default_region",
            "default_instance_type",
            "ssh_key_name",
        ]:
            value = getattr(provider_config, key)
            if value is not None:
                provider_data[key] = value

        if provider_data:
            data[provider] = provider_data

    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(data, f)


def parse_config_key(key: str) -> tuple[str | None, str]:
    """
    Parse a config key into (provider, key) tuple.

    Examples:
        "api_key" -> (None, "api_key")
        "vast.api_key" -> ("vast", "api_key")
        "default_provider" -> (None, "default_provider")
    """
    if "." in key:
        parts = key.split(".", 1)
        return parts[0], parts[1]
    return None, key


def set_config_value(key: str, value: str) -> None:
    """
    Set a single configuration value.

    Supports both flat keys (api_key, default_provider) and
    provider-scoped keys (vast.api_key, prime.default_region).
    """
    provider, actual_key = parse_config_key(key)

    if provider is not None:
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        if actual_key not in PROVIDER_KEYS:
            raise ValueError(f"Unknown config key for provider: {actual_key}")

        data = load_raw_config()
        if provider not in data:
            data[provider] = {}
        data[provider][actual_key] = value
        save_raw_config(data)
    else:
        if actual_key not in TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown config key: {actual_key}")

        if actual_key == "default_provider" and value not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {value}. Must be one of: {', '.join(VALID_PROVIDERS)}"
            )

        data = load_raw_config()
        data[actual_key] = value
        save_raw_config(data)


def get_config_value(key: str) -> str | None:
    """
    Get a single configuration value.

    Supports both flat keys and provider-scoped keys.
    """
    provider, actual_key = parse_config_key(key)

    if provider is not None:
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        if actual_key not in PROVIDER_KEYS:
            raise ValueError(f"Unknown config key for provider: {actual_key}")

        config = load_config()
        provider_config = config.get_provider_config(provider)  # type: ignore[arg-type]
        return getattr(provider_config, actual_key)
    else:
        if actual_key not in TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown config key: {actual_key}")

        config = load_config()
        return getattr(config, actual_key)


def get_provider_api_key(provider: ProviderName) -> str | None:
    """Get the API key for a specific provider."""
    config = load_config()
    provider_config = config.get_provider_config(provider)
    return provider_config.api_key
