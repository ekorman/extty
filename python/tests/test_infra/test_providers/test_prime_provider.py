"""Tests for Prime Intellect provider."""

import pytest
from unittest.mock import patch

from extty_infra.providers.prime_provider import PrimeProvider
from extty_infra.providers.models import ProviderLaunchOptions
from extty_infra.providers.exceptions import ProviderAPIError


@pytest.fixture
def mock_provider():
    """Create a mock Prime provider."""
    with patch.object(PrimeProvider, "_request") as mock_request:
        provider = PrimeProvider("test-api-key")
        provider._mock_request = mock_request
        yield provider


def test_provider_name(mock_provider: PrimeProvider):
    """Test provider name property."""
    assert mock_provider.name == "prime"


def test_ssh_user(mock_provider: PrimeProvider):
    """Test SSH user property."""
    assert mock_provider.ssh_user == "ubuntu"


def test_list_instances(mock_provider: PrimeProvider):
    """Test listing instances."""
    mock_provider._mock_request.return_value = [
        {
            "id": "pod-123",
            "name": "my-pod",
            "ip_address": "10.0.0.1",
            "status": "running",
            "gpu_type": "A100",
            "region": "us-east",
        }
    ]

    instances = mock_provider.list_instances()
    assert len(instances) == 1
    assert instances[0].id == "pod-123"
    assert instances[0].name == "my-pod"
    assert instances[0].ip == "10.0.0.1"
    assert instances[0].status == "running"
    assert instances[0].provider == "prime"
    assert instances[0].ssh_user == "ubuntu"


def test_list_instances_dict_response(mock_provider: PrimeProvider):
    """Test listing instances when API returns dict with pods key."""
    mock_provider._mock_request.return_value = {
        "pods": [
            {
                "id": "pod-123",
                "name": "my-pod",
                "ip_address": "10.0.0.1",
                "status": "running",
                "gpu_type": "A100",
                "region": "us-east",
            }
        ]
    }

    instances = mock_provider.list_instances()
    assert len(instances) == 1
    assert instances[0].id == "pod-123"


def test_get_instance(mock_provider: PrimeProvider):
    """Test getting a specific instance."""
    mock_provider._mock_request.return_value = {
        "pod": {
            "id": "pod-123",
            "name": "my-pod",
            "ip_address": "10.0.0.1",
            "status": "starting",
            "gpu_type": "A100",
            "region": "us-east",
        }
    }

    instance = mock_provider.get_instance("pod-123")
    assert instance.id == "pod-123"
    assert instance.status == "booting"


def test_list_instance_types(mock_provider: PrimeProvider):
    """Test listing instance types."""
    mock_provider._mock_request.return_value = [
        {
            "gpu_type": "A100",
            "gpu_count": 1,
            "description": "NVIDIA A100 80GB",
            "vcpus": 16,
            "memory_gib": 128,
            "storage_gib": 500,
            "price_per_hour": 2.50,
            "regions": ["us-east", "us-west"],
        },
        {
            "gpu_type": "H100",
            "gpu_count": 1,
            "description": "NVIDIA H100 80GB",
            "vcpus": 24,
            "memory_gib": 256,
            "storage_gib": 1000,
            "price_per_hour": 4.00,
            "regions": ["us-east"],
        },
    ]

    types = mock_provider.list_instance_types()
    assert len(types) == 2
    assert types[0].name == "A100x1"
    assert types[0].gpu_count == 1
    assert types[0].gpu_name == "A100"
    assert types[0].price_cents_per_hour == 250
    assert types[1].name == "H100x1"
    assert types[1].price_cents_per_hour == 400


def test_launch(mock_provider: PrimeProvider):
    """Test launching an instance."""
    mock_provider._mock_request.return_value = {"id": "pod-456"}

    options = ProviderLaunchOptions(
        instance_type="A100",
        region="us-east",
        name="my-pod",
    )
    response = mock_provider.launch(options)

    assert response.instance_ids == ["pod-456"]
    mock_provider._mock_request.assert_called_with(
        "POST",
        "/pods/",
        json={
            "gpu_type": "A100",
            "region": "us-east",
            "name": "my-pod",
        },
    )


def test_launch_with_ssh_keys(mock_provider: PrimeProvider):
    """Test launching with SSH keys."""
    mock_provider._mock_request.return_value = {"pod": {"id": "pod-789"}}

    options = ProviderLaunchOptions(
        instance_type="A100",
        ssh_key_names=["my-key"],
    )
    response = mock_provider.launch(options)

    assert response.instance_ids == ["pod-789"]


def test_launch_no_id_returned(mock_provider: PrimeProvider):
    """Test launch fails when no ID returned."""
    mock_provider._mock_request.return_value = {}

    options = ProviderLaunchOptions(instance_type="A100")

    with pytest.raises(ProviderAPIError) as exc:
        mock_provider.launch(options)
    assert "No pod ID returned" in str(exc.value)


def test_terminate(mock_provider: PrimeProvider):
    """Test terminating instances."""
    mock_provider._mock_request.return_value = {}

    mock_provider.terminate(["pod-123"])
    mock_provider._mock_request.assert_called_with("DELETE", "/pods/pod-123")


def test_status_mapping(mock_provider: PrimeProvider):
    """Test status mapping from Prime Intellect to normalized status."""
    assert mock_provider._normalize_status("pending") == "pending"
    assert mock_provider._normalize_status("starting") == "booting"
    assert mock_provider._normalize_status("running") == "running"
    assert mock_provider._normalize_status("stopping") == "stopping"
    assert mock_provider._normalize_status("stopped") == "stopped"
    assert mock_provider._normalize_status("terminated") == "terminated"
    assert mock_provider._normalize_status("failed") == "error"
    assert mock_provider._normalize_status("error") == "error"
    assert mock_provider._normalize_status("unknown") == "error"


def test_context_manager():
    """Test provider can be used as context manager."""
    with patch("httpx.Client"):
        with PrimeProvider("test-key") as provider:
            assert provider.name == "prime"
