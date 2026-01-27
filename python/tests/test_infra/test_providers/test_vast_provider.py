"""Tests for Vast.ai provider."""

import pytest
from unittest.mock import patch

from extty_infra.providers.vast_provider import VastProvider
from extty_infra.providers.models import ProviderLaunchOptions
from extty_infra.providers.exceptions import ProviderAPIError


@pytest.fixture
def mock_provider():
    """Create a mock Vast provider."""
    with patch.object(VastProvider, "_request") as mock_request:
        provider = VastProvider("test-api-key")
        provider._mock_request = mock_request
        yield provider


def test_provider_name(mock_provider: VastProvider):
    """Test provider name property."""
    assert mock_provider.name == "vast"


def test_ssh_user(mock_provider: VastProvider):
    """Test SSH user property."""
    assert mock_provider.ssh_user == "root"


def test_list_instances(mock_provider: VastProvider):
    """Test listing instances."""
    mock_provider._mock_request.return_value = {
        "instances": [
            {
                "id": 12345,
                "label": "my-gpu",
                "ssh_host": "ssh4.vast.ai",
                "ssh_port": 22222,
                "actual_status": "running",
                "gpu_name": "RTX 4090",
                "geolocation": "US",
            }
        ]
    }

    instances = mock_provider.list_instances()
    assert len(instances) == 1
    assert instances[0].id == "12345"
    assert instances[0].name == "my-gpu"
    assert instances[0].ip == "ssh4.vast.ai:22222"
    assert instances[0].status == "running"
    assert instances[0].provider == "vast"
    assert instances[0].ssh_user == "root"


def test_get_instance(mock_provider: VastProvider):
    """Test getting a specific instance."""
    mock_provider._mock_request.return_value = {
        "instances": [
            {
                "id": 12345,
                "label": "my-gpu",
                "ssh_host": "ssh4.vast.ai",
                "ssh_port": 22222,
                "actual_status": "loading",
                "gpu_name": "RTX 4090",
                "geolocation": "US",
            }
        ]
    }

    instance = mock_provider.get_instance("12345")
    assert instance.id == "12345"
    assert instance.status == "booting"


def test_get_instance_not_found(mock_provider: VastProvider):
    """Test getting an instance that doesn't exist."""
    mock_provider._mock_request.return_value = {"instances": []}

    with pytest.raises(ProviderAPIError) as exc:
        mock_provider.get_instance("99999")
    assert "not found" in str(exc.value)


def test_list_instance_types(mock_provider: VastProvider):
    """Test listing instance types."""
    mock_provider._mock_request.return_value = {
        "offers": [
            {
                "id": 1,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "cpu_cores_effective": 16,
                "cpu_ram": 65536,
                "disk_space": 500,
                "dph_base": 0.45,
                "geolocation": "US",
            },
            {
                "id": 2,
                "gpu_name": "RTX 4090",
                "num_gpus": 2,
                "cpu_cores_effective": 32,
                "cpu_ram": 131072,
                "disk_space": 1000,
                "dph_base": 0.45,
                "geolocation": "EU",
            },
        ]
    }

    types = mock_provider.list_instance_types()
    assert len(types) == 2
    assert types[0].name == "RTX 4090x1"
    assert types[0].gpu_count == 1
    assert types[0].gpu_name == "RTX 4090"
    assert types[1].name == "RTX 4090x2"
    assert types[1].gpu_count == 2


def test_launch(mock_provider: VastProvider):
    """Test launching an instance."""
    mock_provider._mock_request.side_effect = [
        {"offers": [{"id": 123}]},
        {"new_contract": 456},
    ]

    options = ProviderLaunchOptions(
        instance_type="RTX 4090",
        name="my-gpu",
    )
    response = mock_provider.launch(options)

    assert response.instance_ids == ["456"]


def test_launch_no_offers(mock_provider: VastProvider):
    """Test launch fails when no offers available."""
    mock_provider._mock_request.return_value = {"offers": []}

    options = ProviderLaunchOptions(instance_type="RTX 4090")

    with pytest.raises(ProviderAPIError) as exc:
        mock_provider.launch(options)
    assert "No offers found" in str(exc.value)


def test_terminate(mock_provider: VastProvider):
    """Test terminating instances."""
    mock_provider._mock_request.return_value = {}

    mock_provider.terminate(["12345"])
    mock_provider._mock_request.assert_called_with("DELETE", "/instances/12345/")


def test_status_mapping(mock_provider: VastProvider):
    """Test status mapping from Vast.ai to normalized status."""
    assert mock_provider._normalize_status("running") == "running"
    assert mock_provider._normalize_status("loading") == "booting"
    assert mock_provider._normalize_status("created") == "pending"
    assert mock_provider._normalize_status("exited") == "stopped"
    assert mock_provider._normalize_status("destroying") == "stopping"
    assert mock_provider._normalize_status("destroyed") == "terminated"
    assert mock_provider._normalize_status("offline") == "error"
    assert mock_provider._normalize_status("unknown") == "error"


def test_context_manager():
    """Test provider can be used as context manager."""
    with patch("httpx.Client"):
        with VastProvider("test-key") as provider:
            assert provider.name == "vast"
