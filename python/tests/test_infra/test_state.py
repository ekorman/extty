"""Tests for state module."""

import pytest
from datetime import datetime
from pathlib import Path

from extty_infra.state import (
    load_state,
    save_state,
    add_instance,
    remove_instance,
    get_instance,
)
from extty_infra.models import InstanceState, InstanceStateFile


@pytest.fixture
def temp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up a temporary state directory."""
    state_dir = tmp_path / ".ex" / "infra"
    state_file = state_dir / "instances.json"

    import extty_infra.state as state_module

    monkeypatch.setattr(state_module, "STATE_DIR", state_dir)
    monkeypatch.setattr(state_module, "STATE_FILE", state_file)

    return state_file


def test_load_state_empty(temp_state: Path):
    """Test loading state when no file exists."""
    state = load_state()
    assert state.instances == []


def test_save_and_load_state(temp_state: Path):
    """Test saving and loading state."""
    instance = InstanceState(
        instance_id="abc123",
        name="test-instance",
        created_at=datetime(2026, 1, 26, 10, 30, 0),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        ip="1.2.3.4",
        status="running",
    )
    state = InstanceStateFile(instances=[instance])
    save_state(state)

    loaded = load_state()
    assert len(loaded.instances) == 1
    assert loaded.instances[0].instance_id == "abc123"
    assert loaded.instances[0].ip == "1.2.3.4"


def test_add_instance(temp_state: Path):
    """Test adding an instance."""
    instance = InstanceState(
        instance_id="test-id",
        created_at=datetime.now(),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        status="booting",
    )
    add_instance(instance)

    state = load_state()
    assert len(state.instances) == 1
    assert state.instances[0].instance_id == "test-id"


def test_add_instance_updates_existing(temp_state: Path):
    """Test that adding an instance with same ID updates it."""
    instance1 = InstanceState(
        instance_id="test-id",
        created_at=datetime.now(),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        status="booting",
    )
    add_instance(instance1)

    instance2 = InstanceState(
        instance_id="test-id",
        created_at=datetime.now(),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        ip="1.2.3.4",
        status="running",
    )
    add_instance(instance2)

    state = load_state()
    assert len(state.instances) == 1
    assert state.instances[0].status == "running"
    assert state.instances[0].ip == "1.2.3.4"


def test_remove_instance(temp_state: Path):
    """Test removing an instance."""
    instance = InstanceState(
        instance_id="test-id",
        created_at=datetime.now(),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        status="running",
    )
    add_instance(instance)
    remove_instance("test-id")

    state = load_state()
    assert len(state.instances) == 0


def test_get_instance(temp_state: Path):
    """Test getting an instance by ID."""
    instance = InstanceState(
        instance_id="test-id",
        name="my-instance",
        created_at=datetime.now(),
        instance_type="gpu_1x_a10",
        region="us-west-1",
        status="running",
    )
    add_instance(instance)

    result = get_instance("test-id")
    assert result is not None
    assert result.name == "my-instance"

    result = get_instance("nonexistent")
    assert result is None
