"""Lambda Cloud provider implementation."""

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

BASE_URL = "https://cloud.lambdalabs.com/api/v1"

LAMBDA_STATUS_MAP: dict[str, NormalizedStatus] = {
    "booting": "booting",
    "active": "running",
    "unhealthy": "error",
    "terminated": "terminated",
    "terminating": "stopping",
    "preempted": "terminated",
}


class LambdaProvider:
    """Lambda Cloud provider implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    @property
    def name(self) -> ProviderName:
        return "lambda"

    @property
    def ssh_user(self) -> str:
        return "ubuntu"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        response = self._client.request(method, path, **kwargs)

        if response.status_code >= 400:
            try:
                error_data = response.json().get("error", {})
                raise ProviderAPIError(
                    code=error_data.get("code", "unknown"),
                    message=error_data.get("message", response.text),
                    suggestion=error_data.get("suggestion"),
                )
            except (ValueError, KeyError):
                response.raise_for_status()

        return response.json()

    def _normalize_status(self, raw_status: str) -> NormalizedStatus:
        return LAMBDA_STATUS_MAP.get(raw_status, "error")

    def list_instances(self) -> list[ProviderInstance]:
        data = self._request("GET", "/instances")
        instances = []
        for item in data.get("data", []):
            raw_status = item.get("status", "unknown")
            instances.append(
                ProviderInstance(
                    id=item["id"],
                    name=item.get("name"),
                    ip=item.get("ip"),
                    status=self._normalize_status(raw_status),
                    instance_type=item.get("instance_type", {}).get("name", "unknown"),
                    region=item.get("region", {}).get("name", "unknown"),
                    provider=self.name,
                    ssh_user=self.ssh_user,
                    raw_status=raw_status,
                )
            )
        return instances

    def get_instance(self, instance_id: str) -> ProviderInstance:
        data = self._request("GET", f"/instances/{instance_id}")
        item = data.get("data", {})
        raw_status = item.get("status", "unknown")
        return ProviderInstance(
            id=item["id"],
            name=item.get("name"),
            ip=item.get("ip"),
            status=self._normalize_status(raw_status),
            instance_type=item.get("instance_type", {}).get("name", "unknown"),
            region=item.get("region", {}).get("name", "unknown"),
            provider=self.name,
            ssh_user=self.ssh_user,
            raw_status=raw_status,
        )

    def list_instance_types(self) -> list[ProviderInstanceType]:
        data = self._request("GET", "/instance-types")
        result = []
        for name, info in data.get("data", {}).items():
            instance_type = info.get("instance_type", {})
            specs = instance_type.get("specs", {})
            regions = [
                r["name"]
                for r in instance_type.get("regions_with_capacity_available", [])
            ]
            result.append(
                ProviderInstanceType(
                    name=name,
                    description=instance_type.get("description"),
                    gpu_count=specs.get("gpus", 0),
                    gpu_name=None,
                    vcpus=specs.get("vcpus", 0),
                    memory_gib=specs.get("memory_gib", 0),
                    storage_gib=specs.get("storage_gib", 0),
                    price_cents_per_hour=instance_type.get("price_cents_per_hour", 0),
                    regions=regions,
                )
            )
        return result

    def launch(self, options: ProviderLaunchOptions) -> ProviderLaunchResponse:
        if not options.region:
            raise ProviderAPIError(
                code="missing_region",
                message="Region is required for Lambda Cloud",
            )
        if not options.ssh_key_names:
            raise ProviderAPIError(
                code="missing_ssh_key",
                message="SSH key name is required for Lambda Cloud",
            )

        payload: dict[str, Any] = {
            "region_name": options.region,
            "instance_type_name": options.instance_type,
            "ssh_key_names": options.ssh_key_names,
        }
        if options.name:
            payload["name"] = options.name

        data = self._request("POST", "/instance-operations/launch", json=payload)
        return ProviderLaunchResponse(
            instance_ids=data.get("data", {}).get("instance_ids", [])
        )

    def terminate(self, instance_ids: list[str]) -> None:
        self._request(
            "POST",
            "/instance-operations/terminate",
            json={"instance_ids": instance_ids},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LambdaProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
