"""Sync local runs to remote server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from extty.storage import get_runs_dir


def push(
    run_name: str,
    server_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """
    Push a local run to the remote server.

    Parameters
    ----------
    run_name : str
        Name of the run to push.
    server_url : str, optional
        Server URL. Defaults to EXTTY_SERVER env var.
    api_key : str, optional
        API key for authentication. Defaults to EXTTY_API_KEY env var.

    Raises
    ------
    ValueError
        If server_url is not provided and EXTTY_SERVER is not set.
    FileNotFoundError
        If the run does not exist locally.
    RuntimeError
        If the push fails.
    """
    server = server_url or os.environ.get("EXTTY_SERVER")
    if not server:
        raise ValueError(
            "Server URL required. Pass server_url or set EXTTY_SERVER env var."
        )

    server = server.rstrip("/")
    key = api_key or os.environ.get("EXTTY_API_KEY")

    run_dir = get_runs_dir() / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run '{run_name}' not found at {run_dir}")

    payload = _build_payload(run_dir)

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = Request(
        f"{server}/api/v1/runs",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(req, timeout=60) as response:
            if response.status == 200 or response.status == 201:
                print(f"Successfully pushed run '{run_name}' to {server}")
            else:
                raise RuntimeError(f"Unexpected response: {response.status}")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Push failed ({e.code}): {body}") from e
    except URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e


def _build_payload(run_dir: Path) -> dict:
    """Build the JSON payload from local run files."""
    meta_path = run_dir / "meta.json"
    metrics_dir = run_dir / "metrics"
    system_path = run_dir / "system.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    payload = {
        "project": meta.get("project", "unknown"),
        "run_name": meta.get("run_name", run_dir.name),
        "config": meta.get("config"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "status": meta.get("status", "completed"),
        "metrics": {},
        "system": [],
    }

    if metrics_dir.exists():
        for csv_file in metrics_dir.rglob("*.csv"):
            metric_name = csv_file.relative_to(metrics_dir).with_suffix("").as_posix()
            points = _parse_metric_csv(csv_file)
            if points:
                payload["metrics"][metric_name] = points

    if system_path.exists():
        payload["system"] = _parse_system_csv(system_path)

    return payload


def _parse_metric_csv(path: Path) -> list[dict]:
    """Parse a metric CSV file into a list of points."""
    points = []
    with open(path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            points.append(
                {
                    "step": int(parts[0]),
                    "timestamp": float(parts[1]),
                    "value": float(parts[2]),
                }
            )

    return points


def _parse_system_csv(path: Path) -> list[dict]:
    """Parse system.csv into a list of system points."""
    points = []
    with open(path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            point = {
                "timestamp": float(parts[0]),
                "ram_used_gb": float(parts[1]) if parts[1] else None,
                "ram_total_gb": float(parts[2]) if parts[2] else None,
                "gpu_mem_used_gb": float(parts[3])
                if len(parts) > 3 and parts[3]
                else None,
                "gpu_mem_total_gb": float(parts[4])
                if len(parts) > 4 and parts[4]
                else None,
                "gpu_util_pct": float(parts[5])
                if len(parts) > 5 and parts[5]
                else None,
            }
            points.append(point)

    return points


def list_local_runs() -> list[str]:
    """List all local run names."""
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    return [
        d.name for d in runs_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()
    ]
