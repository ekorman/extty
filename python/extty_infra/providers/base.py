"""Base protocol for cloud providers."""

from typing import Protocol

from extty_infra.providers.models import (
    ProviderInstance,
    ProviderInstanceType,
    ProviderLaunchOptions,
    ProviderLaunchResponse,
    ProviderName,
)


class CloudProvider(Protocol):
    """Protocol defining the interface for cloud providers."""

    @property
    def name(self) -> ProviderName:
        """Return the provider name."""
        ...

    @property
    def ssh_user(self) -> str:
        """Return the default SSH user for this provider."""
        ...

    def list_instances(self) -> list[ProviderInstance]:
        """List all instances for this provider."""
        ...

    def get_instance(self, instance_id: str) -> ProviderInstance:
        """Get a specific instance by ID."""
        ...

    def list_instance_types(self) -> list[ProviderInstanceType]:
        """List available instance types."""
        ...

    def launch(self, options: ProviderLaunchOptions) -> ProviderLaunchResponse:
        """Launch a new instance."""
        ...

    def terminate(self, instance_ids: list[str]) -> None:
        """Terminate one or more instances."""
        ...

    def close(self) -> None:
        """Close the provider client and release resources."""
        ...
