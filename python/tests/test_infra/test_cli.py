"""Tests for CLI commands."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from extty_infra.cli import app


runner = CliRunner()


@pytest.fixture
def temp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up temporary config and state directories."""
    config_dir = tmp_path / ".ex" / "infra"
    config_file = config_dir / "config.toml"
    state_file = config_dir / "instances.json"

    import extty_infra.config as config_module
    import extty_infra.state as state_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(state_module, "STATE_DIR", config_dir)
    monkeypatch.setattr(state_module, "STATE_FILE", state_file)

    return {"config": config_file, "state": state_file}


def test_config_set(temp_dirs: dict):
    """Test config set command."""
    result = runner.invoke(app, ["config", "set", "api_key", "test-key"])
    assert result.exit_code == 0
    assert "Set api_key" in result.stdout


def test_config_set_provider_key(temp_dirs: dict):
    """Test config set command with provider-scoped key."""
    result = runner.invoke(app, ["config", "set", "vast.api_key", "vast-test-key"])
    assert result.exit_code == 0
    assert "Set vast.api_key" in result.stdout


def test_config_set_default_provider(temp_dirs: dict):
    """Test setting default provider."""
    result = runner.invoke(app, ["config", "set", "default_provider", "vast"])
    assert result.exit_code == 0
    assert "Set default_provider" in result.stdout


def test_config_set_invalid_provider(temp_dirs: dict):
    """Test setting invalid default provider."""
    result = runner.invoke(app, ["config", "set", "default_provider", "invalid"])
    assert result.exit_code == 1
    assert "Invalid provider" in result.stdout


def test_config_show(temp_dirs: dict):
    """Test config show command."""
    runner.invoke(app, ["config", "set", "api_key", "test-key-12345678"])
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "api_key" in result.stdout
    assert "default_provider" in result.stdout


def test_config_set_invalid_key(temp_dirs: dict):
    """Test config set with invalid key."""
    result = runner.invoke(app, ["config", "set", "invalid_key", "value"])
    assert result.exit_code == 1
    assert "Unknown config key" in result.stdout


def test_config_set_invalid_provider_key(temp_dirs: dict):
    """Test config set with invalid provider."""
    result = runner.invoke(app, ["config", "set", "invalid.api_key", "value"])
    assert result.exit_code == 1
    assert "Unknown provider" in result.stdout


def test_types_no_api_key(temp_dirs: dict):
    """Test types command without API key."""
    result = runner.invoke(app, ["types"])
    assert result.exit_code == 1
    assert "API key not set" in result.stdout


def test_types_with_provider_flag(temp_dirs: dict):
    """Test types command with --provider flag."""
    runner.invoke(app, ["config", "set", "vast.api_key", "test-key"])
    with patch(
        "extty_infra.providers.vast_provider.VastProvider.list_instance_types"
    ) as mock:
        mock.return_value = []
        with patch("httpx.Client"):
            result = runner.invoke(app, ["types", "--provider", "vast"])
            assert result.exit_code == 0


def test_launch_missing_params(temp_dirs: dict):
    """Test launch command with missing parameters."""
    runner.invoke(app, ["config", "set", "api_key", "test-key"])
    result = runner.invoke(app, ["launch"])
    assert result.exit_code == 1
    assert (
        "Instance type required" in result.stdout or "Region required" in result.stdout
    )


def test_list_no_api_key(temp_dirs: dict):
    """Test list command without API key."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "API key not set" in result.stdout


@patch("extty_infra.commands.instance.get_provider")
def test_list_empty(mock_get_provider: MagicMock, temp_dirs: dict):
    """Test list command with no instances."""
    runner.invoke(app, ["config", "set", "api_key", "test-key"])

    mock_client = MagicMock()
    mock_client.list_instances.return_value = []
    mock_get_provider.return_value.__enter__.return_value = mock_client

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No instances found" in result.stdout


@patch("extty_infra.commands.instance.get_provider")
def test_list_with_provider_flag(mock_get_provider: MagicMock, temp_dirs: dict):
    """Test list command with --provider flag."""
    runner.invoke(app, ["config", "set", "vast.api_key", "test-key"])

    mock_client = MagicMock()
    mock_client.list_instances.return_value = []
    mock_get_provider.return_value.__enter__.return_value = mock_client

    result = runner.invoke(app, ["list", "--provider", "vast"])
    assert result.exit_code == 0
    assert "No instances found" in result.stdout


def test_help():
    """Test help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Cloud VM Management CLI" in result.stdout


def test_run_missing_path(temp_dirs: dict):
    """Test run command requires path."""
    runner.invoke(app, ["config", "set", "api_key", "test-key"])
    result = runner.invoke(app, ["run", "abc123", "--script", "train.py"])
    assert result.exit_code == 2
    output = result.stdout + (result.output if hasattr(result, "output") else "")
    assert "path" in output.lower() or "missing" in output.lower()


def test_providers_list(temp_dirs: dict):
    """Test providers command lists all providers."""
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "lambda" in result.stdout
    assert "vast" in result.stdout
    assert "prime" in result.stdout
    assert "not configured" in result.stdout


def test_providers_shows_configured(temp_dirs: dict):
    """Test providers command shows configured status."""
    runner.invoke(app, ["config", "set", "vast.api_key", "test-key"])
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "configured" in result.stdout
