"""Prime Intellect provider implementation."""

from typing import Any

import httpx

from extty_infra.providers.exceptions import ProviderAPIError
from extty_infra.providers.models import (
    NormalizedStatus,
    ProviderInstance,
    ProviderInstanceType,
    ProviderLaunchOptions,
    ProviderLaunchResponse,
    ProviderName,
)

BASE_URL = "https://api.primeintellect.ai/api/v1"

PRIME_STATUS_MAP: dict[str, NormalizedStatus] = {
    "pending": "pending",
    "starting": "booting",
    "running": "running",
    "stopping": "stopping",
    "stopped": "stopped",
    "terminated": "terminated",
    "failed": "error",
    "error": "error",
}


class PrimeProvider:
    """Prime Intellect provider implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    @property
    def name(self) -> ProviderName:
        return "prime"

    @property
    def ssh_user(self) -> str:
        return "ubuntu"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        response = self._client.request(method, path, **kwargs)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                msg = (
                    error_data.get("message")
                    or error_data.get("error")
                    or response.text
                )
                raise ProviderAPIError(
                    code=str(response.status_code),
                    message=msg,
                )
            except (ValueError, KeyError):
                response.raise_for_status()

        return response.json()

    def _normalize_status(self, raw_status: str) -> NormalizedStatus:
        return PRIME_STATUS_MAP.get(raw_status.lower(), "error")

    def list_instances(self) -> list[ProviderInstance]:
        data = self._request("GET", "/pods/")
        if not isinstance(data, list):
            data = data.get("pods", []) if isinstance(data, dict) else []

        instances = []
        for item in data:
            raw_status = item.get("status", "unknown")
            instances.append(
                ProviderInstance(
                    id=str(item["id"]),
                    name=item.get("name"),
                    ip=item.get("ip_address") or item.get("ssh_host"),
                    status=self._normalize_status(raw_status),
                    instance_type=item.get("gpu_type", "unknown"),
                    region=item.get("region", "unknown"),
                    provider=self.name,
                    ssh_user=self.ssh_user,
                    raw_status=raw_status,
                )
            )
        return instances

    def get_instance(self, instance_id: str) -> ProviderInstance:
        data = self._request("GET", f"/pods/{instance_id}")
        if not isinstance(data, dict):
            raise ProviderAPIError(
                code="not_found",
                message=f"Pod {instance_id} not found",
            )

        item = data.get("pod", data)
        raw_status = item.get("status", "unknown")

        return ProviderInstance(
            id=str(item["id"]),
            name=item.get("name"),
            ip=item.get("ip_address") or item.get("ssh_host"),
            status=self._normalize_status(raw_status),
            instance_type=item.get("gpu_type", "unknown"),
            region=item.get("region", "unknown"),
            provider=self.name,
            ssh_user=self.ssh_user,
            raw_status=raw_status,
        )

    def list_instance_types(self) -> list[ProviderInstanceType]:
        data = self._request("GET", "/availability/gpus")
        if not isinstance(data, list):
            data = data.get("gpus", []) if isinstance(data, dict) else []

        result = []
        for item in data:
            gpu_type = item.get("gpu_type", "unknown")
            gpu_count = item.get("gpu_count", 1)

            result.append(
                ProviderInstanceType(
                    name=f"{gpu_type}x{gpu_count}",
                    description=item.get("description"),
                    gpu_count=gpu_count,
                    gpu_name=gpu_type,
                    vcpus=item.get("vcpus", 0),
                    memory_gib=item.get("memory_gib", 0),
                    storage_gib=item.get("storage_gib", 0),
                    price_cents_per_hour=int(item.get("price_per_hour", 0) * 100),
                    regions=item.get("regions", []),
                )
            )

        return result

    def launch(self, options: ProviderLaunchOptions) -> ProviderLaunchResponse:
        payload: dict[str, Any] = {
            "gpu_type": options.instance_type,
        }
        if options.region:
            payload["region"] = options.region
        if options.name:
            payload["name"] = options.name
        if options.ssh_key_names:
            payload["ssh_key_names"] = options.ssh_key_names

        data = self._request("POST", "/pods/", json=payload)
        if not isinstance(data, dict):
            raise ProviderAPIError(
                code="launch_failed",
                message="Failed to create pod",
            )

        pod_id = data.get("id") or data.get("pod", {}).get("id")
        if not pod_id:
            raise ProviderAPIError(
                code="launch_failed",
                message="No pod ID returned from Prime Intellect",
            )

        return ProviderLaunchResponse(instance_ids=[str(pod_id)])

    def terminate(self, instance_ids: list[str]) -> None:
        for instance_id in instance_ids:
            self._request("DELETE", f"/pods/{instance_id}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PrimeProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
