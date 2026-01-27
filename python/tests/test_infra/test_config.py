"""Tests for config module."""

import pytest
from pathlib import Path

from extty_infra.config import (
    load_config,
    save_config,
    set_config_value,
    get_config_value,
)
from extty_infra.models import InfraConfig


@pytest.fixture
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a temporary config directory."""
    config_dir = tmp_path / ".ex" / "infra"
    config_file = config_dir / "config.toml"

    import extty_infra.config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)

    return config_file


def test_load_config_empty(temp_config: Path):
    """Test loading config when no file exists."""
    config = load_config()
    assert config.api_key is None
    assert config.default_region is None


def test_save_and_load_config(temp_config: Path):
    """Test saving and loading config."""
    config = InfraConfig(
        api_key="test-key",
        default_region="us-west-1",
        default_instance_type="gpu_1x_a10",
    )
    save_config(config)

    loaded = load_config()
    assert loaded.api_key == "test-key"
    assert loaded.default_region == "us-west-1"
    assert loaded.default_instance_type == "gpu_1x_a10"


def test_set_config_value(temp_config: Path):
    """Test setting individual config values."""
    set_config_value("api_key", "my-api-key")
    assert get_config_value("api_key") == "my-api-key"

    set_config_value("default_region", "us-east-1")
    assert get_config_value("default_region") == "us-east-1"


def test_set_config_invalid_key(temp_config: Path):
    """Test setting invalid config key raises error."""
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("invalid_key", "value")


def test_get_config_invalid_key(temp_config: Path):
    """Test getting invalid config key raises error."""
    with pytest.raises(ValueError, match="Unknown config key"):
        get_config_value("invalid_key")
