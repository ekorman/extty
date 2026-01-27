"""Normalized models for cloud providers."""

from typing import Literal

from pydantic import BaseModel

ProviderName = Literal["lambda", "vast", "prime"]

NormalizedStatus = Literal[
    "pending", "booting", "running", "stopping", "stopped", "terminated", "error"
]


class ProviderInstance(BaseModel):
    """Normalized instance representation across providers."""

    id: str
    name: str | None
    ip: str | None
    status: NormalizedStatus
    instance_type: str
    region: str
    provider: ProviderName
    ssh_user: str
    raw_status: str


class ProviderInstanceType(BaseModel):
    """Normalized instance type representation across providers."""

    name: str
    description: str | None
    gpu_count: int
    gpu_name: str | None
    vcpus: int
    memory_gib: int
    storage_gib: int
    price_cents_per_hour: int
    regions: list[str]


class ProviderLaunchOptions(BaseModel):
    """Options for launching an instance on any provider."""

    instance_type: str
    region: str | None = None
    ssh_key_names: list[str] | None = None
    name: str | None = None


class ProviderLaunchResponse(BaseModel):
    """Response from launching an instance."""

    instance_ids: list[str]
