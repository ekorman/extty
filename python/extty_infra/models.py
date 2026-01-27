"""Pydantic models for cloud providers, config, and state."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from extty_infra.providers.models import NormalizedStatus, ProviderName

InstanceStatus = Literal[
    "booting", "active", "unhealthy", "terminated", "terminating", "preempted"
]


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    api_key: str | None = None
    default_region: str | None = None
    default_instance_type: str | None = None
    ssh_key_name: str | None = None


class InfraConfig(BaseModel):
    """Configuration stored in ~/.ex/infra/config.toml."""

    default_provider: ProviderName = "lambda"

    # Per-provider configs
    lambda_config: ProviderConfig = Field(
        default_factory=ProviderConfig, alias="lambda"
    )
    vast: ProviderConfig = Field(default_factory=ProviderConfig)
    prime: ProviderConfig = Field(default_factory=ProviderConfig)

    # Legacy top-level keys for backward compatibility with existing configs
    api_key: str | None = None
    default_region: str | None = None
    default_instance_type: str | None = None
    ssh_key_name: str | None = None

    model_config = {"populate_by_name": True}

    def get_provider_config(self, provider: ProviderName) -> ProviderConfig:
        """Get config for a specific provider, with legacy fallback for lambda."""
        if provider == "lambda":
            config = self.lambda_config
            return ProviderConfig(
                api_key=config.api_key or self.api_key,
                default_region=config.default_region or self.default_region,
                default_instance_type=config.default_instance_type
                or self.default_instance_type,
                ssh_key_name=config.ssh_key_name or self.ssh_key_name,
            )
        elif provider == "vast":
            return self.vast
        elif provider == "prime":
            return self.prime
        else:
            raise ValueError(f"Unknown provider: {provider}")


class InstanceState(BaseModel):
    """State for a single tracked instance."""

    instance_id: str
    provider: ProviderName = "lambda"
    name: str | None = None
    created_at: datetime
    instance_type: str
    region: str
    ip: str | None = None
    status: NormalizedStatus
    ssh_user: str = "ubuntu"


class InstanceStateFile(BaseModel):
    """State file stored in ~/.ex/infra/instances.json."""

    instances: list[InstanceState] = Field(default_factory=list)


class Region(BaseModel):
    """Lambda Cloud region."""

    name: str
    description: str


class InstanceTypeSpec(BaseModel):
    """Hardware specs for an instance type."""

    vcpus: int
    memory_gib: int
    storage_gib: int
    gpus: int


class InstanceTypePrice(BaseModel):
    """Pricing for an instance type in a region."""

    cents_per_hour: int


class RegionAvailability(BaseModel):
    """Availability info for a region."""

    name: str
    description: str


class InstanceType(BaseModel):
    """Lambda Cloud instance type."""

    name: str
    description: str
    specs: InstanceTypeSpec
    price_cents_per_hour: int
    regions_with_capacity_available: list[RegionAvailability]


class SSHKey(BaseModel):
    """Lambda Cloud SSH key."""

    id: str
    name: str
    public_key: str


class Instance(BaseModel):
    """Lambda Cloud instance from API."""

    id: str
    name: str | None = None
    ip: str | None = None
    status: InstanceStatus
    ssh_key_names: list[str]
    file_system_names: list[str]
    region: Region
    instance_type: InstanceType
    hostname: str | None = None
    jupyter_token: str | None = None
    jupyter_url: str | None = None


class LaunchOptions(BaseModel):
    """Options for launching an instance."""

    region_name: str
    instance_type_name: str
    ssh_key_names: list[str]
    name: str | None = None


class RunOptions(BaseModel):
    """Options for the run command."""

    instance_id: str
    local_path: str
    script: str
    args: str | None = None


class LaunchResponse(BaseModel):
    """Response from launching instances."""

    instance_ids: list[str]


class TerminateResponse(BaseModel):
    """Response from terminating instances."""

    terminated_instances: list[dict]


class APIError(BaseModel):
    """Lambda Cloud API error response."""

    code: str
    message: str
    suggestion: str | None = None
