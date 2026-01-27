"""Tests for Lambda Cloud provider."""

import pytest
from unittest.mock import patch

from extty_infra.providers.lambda_provider import LambdaProvider
from extty_infra.providers.models import ProviderLaunchOptions
from extty_infra.providers.exceptions import ProviderAPIError


@pytest.fixture
def mock_provider():
    """Create a mock Lambda provider."""
    with patch.object(LambdaProvider, "_request") as mock_request:
        provider = LambdaProvider("test-api-key")
        provider._mock_request = mock_request
        yield provider


def test_provider_name(mock_provider: LambdaProvider):
    """Test provider name property."""
    assert mock_provider.name == "lambda"


def test_ssh_user(mock_provider: LambdaProvider):
    """Test SSH user property."""
    assert mock_provider.ssh_user == "ubuntu"


def test_list_instances(mock_provider: LambdaProvider):
    """Test listing instances."""
    mock_provider._mock_request.return_value = {
        "data": [
            {
                "id": "abc123",
                "name": "test-vm",
                "ip": "1.2.3.4",
                "status": "active",
                "instance_type": {"name": "gpu_1x_a10"},
                "region": {"name": "us-west-1"},
            }
        ]
    }

    instances = mock_provider.list_instances()
    assert len(instances) == 1
    assert instances[0].id == "abc123"
    assert instances[0].status == "running"
    assert instances[0].provider == "lambda"
    assert instances[0].ssh_user == "ubuntu"


def test_get_instance(mock_provider: LambdaProvider):
    """Test getting a specific instance."""
    mock_provider._mock_request.return_value = {
        "data": {
            "id": "abc123",
            "name": "test-vm",
            "ip": "1.2.3.4",
            "status": "booting",
            "instance_type": {"name": "gpu_1x_a10"},
            "region": {"name": "us-west-1"},
        }
    }

    instance = mock_provider.get_instance("abc123")
    assert instance.id == "abc123"
    assert instance.status == "booting"


def test_list_instance_types(mock_provider: LambdaProvider):
    """Test listing instance types."""
    mock_provider._mock_request.return_value = {
        "data": {
            "gpu_1x_a10": {
                "instance_type": {
                    "description": "1x A10 GPU",
                    "specs": {
                        "vcpus": 30,
                        "memory_gib": 200,
                        "storage_gib": 1400,
                        "gpus": 1,
                    },
                    "price_cents_per_hour": 60,
                    "regions_with_capacity_available": [
                        {"name": "us-west-1", "description": "US West 1"}
                    ],
                }
            }
        }
    }

    types = mock_provider.list_instance_types()
    assert len(types) == 1
    assert types[0].name == "gpu_1x_a10"
    assert types[0].gpu_count == 1
    assert types[0].vcpus == 30
    assert types[0].regions == ["us-west-1"]


def test_launch(mock_provider: LambdaProvider):
    """Test launching an instance."""
    mock_provider._mock_request.return_value = {"data": {"instance_ids": ["abc123"]}}

    options = ProviderLaunchOptions(
        instance_type="gpu_1x_a10",
        region="us-west-1",
        ssh_key_names=["my-key"],
        name="test-vm",
    )
    response = mock_provider.launch(options)

    assert response.instance_ids == ["abc123"]
    mock_provider._mock_request.assert_called_with(
        "POST",
        "/instance-operations/launch",
        json={
            "region_name": "us-west-1",
            "instance_type_name": "gpu_1x_a10",
            "ssh_key_names": ["my-key"],
            "name": "test-vm",
        },
    )


def test_launch_missing_region(mock_provider: LambdaProvider):
    """Test launch fails without region."""
    options = ProviderLaunchOptions(
        instance_type="gpu_1x_a10",
        ssh_key_names=["my-key"],
    )

    with pytest.raises(ProviderAPIError) as exc:
        mock_provider.launch(options)
    assert "Region is required" in str(exc.value)


def test_launch_missing_ssh_key(mock_provider: LambdaProvider):
    """Test launch fails without SSH key."""
    options = ProviderLaunchOptions(
        instance_type="gpu_1x_a10",
        region="us-west-1",
    )

    with pytest.raises(ProviderAPIError) as exc:
        mock_provider.launch(options)
    assert "SSH key name is required" in str(exc.value)


def test_terminate(mock_provider: LambdaProvider):
    """Test terminating instances."""
    mock_provider._mock_request.return_value = {
        "data": {"terminated_instances": [{"id": "abc123"}]}
    }

    mock_provider.terminate(["abc123"])
    mock_provider._mock_request.assert_called_with(
        "POST",
        "/instance-operations/terminate",
        json={"instance_ids": ["abc123"]},
    )


def test_status_mapping(mock_provider: LambdaProvider):
    """Test status mapping from Lambda to normalized status."""
    assert mock_provider._normalize_status("booting") == "booting"
    assert mock_provider._normalize_status("active") == "running"
    assert mock_provider._normalize_status("unhealthy") == "error"
    assert mock_provider._normalize_status("terminated") == "terminated"
    assert mock_provider._normalize_status("terminating") == "stopping"
    assert mock_provider._normalize_status("preempted") == "terminated"
    assert mock_provider._normalize_status("unknown") == "error"


def test_context_manager():
    """Test provider can be used as context manager."""
    with patch("httpx.Client"):
        with LambdaProvider("test-key") as provider:
            assert provider.name == "lambda"
