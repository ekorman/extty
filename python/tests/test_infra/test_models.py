"""Tests for models module."""

from datetime import datetime

from extty_infra.models import (
    InfraConfig,
    InstanceState,
    InstanceStateFile,
    LaunchOptions,
    Instance,
    InstanceType,
    InstanceTypeSpec,
    Region,
    RegionAvailability,
)


def test_infra_config_defaults():
    """Test InfraConfig with defaults."""
    config = InfraConfig()
    assert config.api_key is None
    assert config.default_region is None
    assert config.default_instance_type is None
    assert config.ssh_key_name is None


def test_infra_config_with_values():
    """Test InfraConfig with values."""
    config = InfraConfig(
        api_key="test-key",
        default_region="us-west-1",
        default_instance_type="gpu_1x_a10",
        ssh_key_name="my-key",
    )
    assert config.api_key == "test-key"
    assert config.default_region == "us-west-1"


def test_instance_state():
    """Test InstanceState model."""
    state = InstanceState(
        instance_id="abc123",
        name="test-vm",
        created_at=datetime(2026, 1, 26, 10, 30, 0),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        ip="1.2.3.4",
        status="running",
    )
    assert state.instance_id == "abc123"
    assert state.provider == "lambda"
    assert state.status == "running"


def test_instance_state_file():
    """Test InstanceStateFile model."""
    state = InstanceStateFile()
    assert state.instances == []

    state = InstanceStateFile(
        instances=[
            InstanceState(
                instance_id="abc123",
                created_at=datetime.now(),
                instance_type="gpu_1x_a10",
                region="us-west-1",
                status="running",
            )
        ]
    )
    assert len(state.instances) == 1


def test_launch_options():
    """Test LaunchOptions model."""
    options = LaunchOptions(
        region_name="us-west-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_names=["my-key"],
        name="test-vm",
    )
    assert options.region_name == "us-west-1"
    assert options.ssh_key_names == ["my-key"]


def test_instance_type():
    """Test InstanceType model."""
    it = InstanceType(
        name="gpu_1x_a10",
        description="1x A10 GPU",
        specs=InstanceTypeSpec(
            vcpus=30,
            memory_gib=200,
            storage_gib=1400,
            gpus=1,
        ),
        price_cents_per_hour=60,
        regions_with_capacity_available=[
            RegionAvailability(name="us-west-1", description="US West 1")
        ],
    )
    assert it.name == "gpu_1x_a10"
    assert it.specs.gpus == 1
    assert it.price_cents_per_hour == 60


def test_instance():
    """Test Instance model from API."""
    instance = Instance(
        id="abc123",
        name="test-vm",
        ip="1.2.3.4",
        status="active",
        ssh_key_names=["my-key"],
        file_system_names=[],
        region=Region(name="us-west-1", description="US West 1"),
        instance_type=InstanceType(
            name="gpu_1x_a10",
            description="1x A10 GPU",
            specs=InstanceTypeSpec(vcpus=30, memory_gib=200, storage_gib=1400, gpus=1),
            price_cents_per_hour=60,
            regions_with_capacity_available=[],
        ),
    )
    assert instance.id == "abc123"
    assert instance.status == "active"
    assert instance.region.name == "us-west-1"
