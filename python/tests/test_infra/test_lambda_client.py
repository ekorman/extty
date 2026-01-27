"""Tests for Lambda client module."""

import pytest
from unittest.mock import patch

from extty_infra.lambda_client import LambdaClient
from extty_infra.models import LaunchOptions


@pytest.fixture
def mock_client():
    """Create a mock Lambda client."""
    with patch.object(LambdaClient, "_request") as mock_request:
        client = LambdaClient("test-api-key")
        client._mock_request = mock_request
        yield client


def test_list_instances(mock_client: LambdaClient):
    """Test listing instances."""
    mock_client._mock_request.return_value = {
        "data": [
            {
                "id": "abc123",
                "name": "test-vm",
                "ip": "1.2.3.4",
                "status": "active",
                "ssh_key_names": ["my-key"],
                "file_system_names": [],
                "region": {"name": "us-west-1", "description": "US West 1"},
                "instance_type": {
                    "name": "gpu_1x_a10",
                    "description": "1x A10",
                    "specs": {
                        "vcpus": 30,
                        "memory_gib": 200,
                        "storage_gib": 1400,
                        "gpus": 1,
                    },
                    "price_cents_per_hour": 60,
                    "regions_with_capacity_available": [],
                },
            }
        ]
    }

    instances = mock_client.list_instances()
    assert len(instances) == 1
    assert instances[0].id == "abc123"
    assert instances[0].status == "active"


def test_list_instance_types(mock_client: LambdaClient):
    """Test listing instance types."""
    mock_client._mock_request.return_value = {
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

    types = mock_client.list_instance_types()
    assert len(types) == 1
    assert types[0].name == "gpu_1x_a10"
    assert types[0].specs.gpus == 1


def test_launch(mock_client: LambdaClient):
    """Test launching an instance."""
    mock_client._mock_request.return_value = {"data": {"instance_ids": ["abc123"]}}

    options = LaunchOptions(
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_names=["my-key"],
        name="test-vm",
    )
    response = mock_client.launch(options)

    assert response.instance_ids == ["abc123"]
    mock_client._mock_request.assert_called_with(
        "POST",
        "/instance-operations/launch",
        json={
            "region_name": "us-west-1",
            "instance_type_name": "gpu_1x_a10",
            "ssh_key_names": ["my-key"],
            "name": "test-vm",
        },
    )


def test_terminate(mock_client: LambdaClient):
    """Test terminating instances."""
    mock_client._mock_request.return_value = {
        "data": {"terminated_instances": [{"id": "abc123"}]}
    }

    response = mock_client.terminate(["abc123"])
    assert len(response.terminated_instances) == 1


def test_list_ssh_keys(mock_client: LambdaClient):
    """Test listing SSH keys."""
    mock_client._mock_request.return_value = {
        "data": [{"id": "key-1", "name": "my-key", "public_key": "ssh-rsa AAAA..."}]
    }

    keys = mock_client.list_ssh_keys()
    assert len(keys) == 1
    assert keys[0].name == "my-key"


def test_client_context_manager():
    """Test client can be used as context manager."""
    with patch("httpx.Client"):
        with LambdaClient("test-key") as client:
            assert client.api_key == "test-key"
