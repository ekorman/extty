"""Vast.ai provider implementation."""

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

BASE_URL = "https://console.vast.ai/api/v0"

VAST_STATUS_MAP: dict[str, NormalizedStatus] = {
    "running": "running",
    "loading": "booting",
    "created": "pending",
    "exited": "stopped",
    "destroying": "stopping",
    "destroyed": "terminated",
    "offline": "error",
}


class VastProvider:
    """Vast.ai provider implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    @property
    def name(self) -> ProviderName:
        return "vast"

    @property
    def ssh_user(self) -> str:
        return "root"

    def _request(
        self, method: str, path: str, params: dict | None = None, **kwargs: Any
    ) -> dict | list:
        response = self._client.request(method, path, params=params, **kwargs)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                msg = error_data.get("msg") or error_data.get("error") or response.text
                raise ProviderAPIError(
                    code=str(response.status_code),
                    message=msg,
                )
            except (ValueError, KeyError):
                response.raise_for_status()

        return response.json()

    def _normalize_status(self, raw_status: str) -> NormalizedStatus:
        return VAST_STATUS_MAP.get(raw_status.lower(), "error")

    def list_instances(self) -> list[ProviderInstance]:
        data = self._request("GET", "/instances/")
        if not isinstance(data, dict):
            return []

        instances = []
        for item in data.get("instances", []):
            raw_status = item.get("actual_status", item.get("status_msg", "unknown"))
            ssh_port = item.get("ssh_port", 22)
            ssh_host = item.get("ssh_host")
            ip = f"{ssh_host}:{ssh_port}" if ssh_host else None

            instances.append(
                ProviderInstance(
                    id=str(item["id"]),
                    name=item.get("label"),
                    ip=ip,
                    status=self._normalize_status(raw_status),
                    instance_type=item.get("gpu_name", "unknown"),
                    region=item.get("geolocation", "unknown"),
                    provider=self.name,
                    ssh_user=self.ssh_user,
                    raw_status=raw_status,
                )
            )
        return instances

    def get_instance(self, instance_id: str) -> ProviderInstance:
        data = self._request("GET", "/instances/", params={"id": instance_id})
        if not isinstance(data, dict):
            raise ProviderAPIError(
                code="not_found",
                message=f"Instance {instance_id} not found",
            )

        instances = data.get("instances", [])
        for item in instances:
            if str(item.get("id")) == instance_id:
                raw_status = item.get(
                    "actual_status", item.get("status_msg", "unknown")
                )
                ssh_port = item.get("ssh_port", 22)
                ssh_host = item.get("ssh_host")
                ip = f"{ssh_host}:{ssh_port}" if ssh_host else None

                return ProviderInstance(
                    id=str(item["id"]),
                    name=item.get("label"),
                    ip=ip,
                    status=self._normalize_status(raw_status),
                    instance_type=item.get("gpu_name", "unknown"),
                    region=item.get("geolocation", "unknown"),
                    provider=self.name,
                    ssh_user=self.ssh_user,
                    raw_status=raw_status,
                )

        raise ProviderAPIError(
            code="not_found",
            message=f"Instance {instance_id} not found",
        )

    def list_instance_types(self) -> list[ProviderInstanceType]:
        payload = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "num_gpus": {"gte": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 100,
        }
        data = self._request("POST", "/bundles/", json=payload)
        if not isinstance(data, dict):
            return []

        result = []
        seen_types: set[str] = set()

        for offer in data.get("offers", []):
            gpu_name = offer.get("gpu_name", "unknown")
            num_gpus = offer.get("num_gpus", 1)
            type_key = f"{gpu_name}x{num_gpus}"

            if type_key in seen_types:
                continue
            seen_types.add(type_key)

            price_per_gpu = offer.get("dph_base", 0)
            total_price = price_per_gpu * num_gpus

            result.append(
                ProviderInstanceType(
                    name=type_key,
                    description=f"{num_gpus}x {gpu_name}",
                    gpu_count=num_gpus,
                    gpu_name=gpu_name,
                    vcpus=offer.get("cpu_cores_effective", 0),
                    memory_gib=int(offer.get("cpu_ram", 0) / 1024),
                    storage_gib=int(offer.get("disk_space", 0)),
                    price_cents_per_hour=int(total_price * 100),
                    regions=[offer.get("geolocation", "unknown")],
                )
            )

        return result

    def launch(self, options: ProviderLaunchOptions) -> ProviderLaunchResponse:
        search_payload = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "gpu_name": {"eq": options.instance_type},
            "num_gpus": {"gte": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 1,
        }

        search_data = self._request("POST", "/bundles/", json=search_payload)
        if not isinstance(search_data, dict):
            raise ProviderAPIError(
                code="no_offers",
                message=f"No offers found for {options.instance_type}",
            )

        offers = search_data.get("offers", [])
        if not offers:
            raise ProviderAPIError(
                code="no_offers",
                message=f"No offers found for {options.instance_type}",
            )

        offer = offers[0]
        offer_id = offer.get("id")

        create_payload: dict[str, Any] = {
            "client_id": "me",
            "image": "pytorch/pytorch:latest",
            "disk": 50,
            "onstart": "",
        }
        if options.name:
            create_payload["label"] = options.name

        data = self._request("PUT", f"/asks/{offer_id}/", json=create_payload)
        if not isinstance(data, dict):
            raise ProviderAPIError(
                code="launch_failed",
                message="Failed to create instance",
            )

        new_contract = data.get("new_contract")
        if not new_contract:
            raise ProviderAPIError(
                code="launch_failed",
                message="No contract returned from Vast.ai",
            )

        return ProviderLaunchResponse(instance_ids=[str(new_contract)])

    def terminate(self, instance_ids: list[str]) -> None:
        for instance_id in instance_ids:
            self._request("DELETE", f"/instances/{instance_id}/")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VastProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
