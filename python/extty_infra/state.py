"""State file handling for ~/.ex/infra/instances.json."""

import json
from datetime import datetime
from pathlib import Path

from extty_infra.models import InstanceState, InstanceStateFile
from extty_infra.providers.models import ProviderInstance, ProviderName

STATE_DIR = Path.home() / ".ex" / "infra"
STATE_FILE = STATE_DIR / "instances.json"


def ensure_state_dir() -> None:
    """Create state directory if it doesn't exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> InstanceStateFile:
    """Load state from JSON file."""
    if not STATE_FILE.exists():
        return InstanceStateFile()

    with open(STATE_FILE) as f:
        data = json.load(f)
    return InstanceStateFile.model_validate(data)


def save_state(state: InstanceStateFile) -> None:
    """Save state to JSON file."""
    ensure_state_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2, default=str)


def add_instance(instance_state: InstanceState) -> None:
    """Add an instance to the state file."""
    state = load_state()
    state.instances = [
        i for i in state.instances if i.instance_id != instance_state.instance_id
    ]
    state.instances.append(instance_state)
    save_state(state)


def remove_instance(instance_id: str) -> None:
    """Remove an instance from the state file."""
    state = load_state()
    state.instances = [i for i in state.instances if i.instance_id != instance_id]
    save_state(state)


def get_instance(instance_id: str) -> InstanceState | None:
    """Get an instance by ID from the state file."""
    state = load_state()
    for instance in state.instances:
        if instance.instance_id == instance_id:
            return instance
    return None


def get_instances_by_provider(provider: ProviderName) -> list[InstanceState]:
    """Get all instances for a specific provider."""
    state = load_state()
    return [i for i in state.instances if i.provider == provider]


def update_instance_from_provider(provider_instance: ProviderInstance) -> InstanceState:
    """Update local state from provider instance data."""
    state = load_state()
    existing = next(
        (i for i in state.instances if i.instance_id == provider_instance.id), None
    )

    instance_state = InstanceState(
        instance_id=provider_instance.id,
        provider=provider_instance.provider,
        name=provider_instance.name,
        created_at=existing.created_at if existing else datetime.now(),
        instance_type=provider_instance.instance_type,
        region=provider_instance.region,
        ip=provider_instance.ip,
        status=provider_instance.status,
        ssh_user=provider_instance.ssh_user,
    )

    add_instance(instance_state)
    return instance_state


def sync_with_provider(
    provider: ProviderName, api_instances: list[ProviderInstance]
) -> list[InstanceState]:
    """Sync local state with provider instances for a specific provider."""
    state = load_state()
    api_ids = {i.id for i in api_instances}

    for instance in state.instances:
        if instance.provider != provider:
            continue
        if instance.instance_id not in api_ids:
            if instance.status not in ("terminated", "stopping"):
                instance.status = "terminated"

    save_state(state)

    for api_instance in api_instances:
        update_instance_from_provider(api_instance)

    updated_state = load_state()
    return [i for i in updated_state.instances if i.provider == provider]


def sync_all_providers(
    provider_instances: dict[ProviderName, list[ProviderInstance]],
) -> list[InstanceState]:
    """Sync local state with instances from multiple providers."""
    all_instances: list[InstanceState] = []

    for provider, instances in provider_instances.items():
        synced = sync_with_provider(provider, instances)
        all_instances.extend(synced)

    return all_instances
