"""Custom Lambda Cloud API client using httpx."""

from typing import Any

import httpx

from extty_infra.models import (
    APIError,
    Instance,
    InstanceType,
    LaunchOptions,
    LaunchResponse,
    SSHKey,
    TerminateResponse,
)

BASE_URL = "https://cloud.lambdalabs.com/api/v1"


class LambdaCloudError(Exception):
    """Exception raised for Lambda Cloud API errors."""

    def __init__(self, error: APIError):
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class LambdaClient:
    """Client for Lambda Cloud API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Make an API request and handle errors."""
        response = self._client.request(method, path, **kwargs)

        if response.status_code >= 400:
            try:
                error_data = response.json().get("error", {})
                error = APIError.model_validate(error_data)
                raise LambdaCloudError(error)
            except (ValueError, KeyError):
                response.raise_for_status()

        return response.json()

    def list_instances(self) -> list[Instance]:
        """List all running instances."""
        data = self._request("GET", "/instances")
        return [Instance.model_validate(i) for i in data.get("data", [])]

    def get_instance(self, instance_id: str) -> Instance:
        """Get details for a specific instance."""
        data = self._request("GET", f"/instances/{instance_id}")
        return Instance.model_validate(data.get("data", {}))

    def list_instance_types(self) -> list[InstanceType]:
        """List available instance types."""
        data = self._request("GET", "/instance-types")
        result = []
        for name, info in data.get("data", {}).items():
            instance_type = info.get("instance_type", {})
            instance_type["name"] = name
            result.append(InstanceType.model_validate(instance_type))
        return result

    def launch(self, options: LaunchOptions) -> LaunchResponse:
        """Launch a new instance."""
        payload = {
            "region_name": options.region_name,
            "instance_type_name": options.instance_type_name,
            "ssh_key_names": options.ssh_key_names,
        }
        if options.name:
            payload["name"] = options.name

        data = self._request("POST", "/instance-operations/launch", json=payload)
        return LaunchResponse.model_validate(data.get("data", {}))

    def terminate(self, instance_ids: list[str]) -> TerminateResponse:
        """Terminate instances."""
        data = self._request(
            "POST",
            "/instance-operations/terminate",
            json={"instance_ids": instance_ids},
        )
        return TerminateResponse.model_validate(data.get("data", {}))

    def list_ssh_keys(self) -> list[SSHKey]:
        """List SSH keys."""
        data = self._request("GET", "/ssh-keys")
        return [SSHKey.model_validate(k) for k in data.get("data", [])]

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "LambdaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
