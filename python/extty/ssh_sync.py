"""SSH synchronization for reading runs from remote servers."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

try:
    import paramiko
except ImportError:
    paramiko = None


def _ensure_paramiko() -> None:
    """Ensure paramiko is installed."""
    if paramiko is None:
        raise ImportError(
            "paramiko is required for SSH support. "
            "Install it with: pip install extty[ssh]"
        )


def parse_ssh_host(server: str) -> tuple[str, str]:
    """
    Parse SSH server string into username and host.

    Parameters
    ----------
    server : str
        Server string in format "user@host" or just "host".

    Returns
    -------
    tuple[str, str]
        Username and hostname.

    Raises
    ------
    ValueError
        If server format is invalid.
    """
    if "@" in server:
        parts = server.split("@", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid server format: {server}")
        return parts[0], parts[1]
    else:
        # Use current user if no username specified
        import getpass
        return getpass.getuser(), server


def create_ssh_client(username: str, hostname: str) -> paramiko.SSHClient:
    """
    Create and connect an SSH client.

    Parameters
    ----------
    username : str
        SSH username.
    hostname : str
        SSH hostname.

    Returns
    -------
    paramiko.SSHClient
        Connected SSH client.

    Raises
    ------
    Exception
        If connection fails.
    """
    _ensure_paramiko()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Try to connect using SSH agent or local keys
        client.connect(
            hostname,
            username=username,
            look_for_keys=True,
            allow_agent=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to {username}@{hostname}: {e}"
        ) from e

    return client


def list_remote_runs(server: str) -> list[str]:
    """
    List all runs on a remote server via SSH.

    Parameters
    ----------
    server : str
        Server in format "user@host" or "host".

    Returns
    -------
    list[str]
        List of run names.

    Raises
    ------
    RuntimeError
        If SSH connection or command execution fails.
    """
    _ensure_paramiko()
    username, hostname = parse_ssh_host(server)

    client = create_ssh_client(username, hostname)
    try:
        # List directories in ~/.ex/runs/ that have meta.json
        cmd = (
            'cd ~/.ex/runs 2>/dev/null && '
            'for d in */; do '
            '  if [ -f "$d/meta.json" ]; then '
            '    basename "$d"; '
            '  fi; '
            'done'
        )

        stdin, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            stderr_text = stderr.read().decode('utf-8')
            if "No such file or directory" in stderr_text:
                return []
            raise RuntimeError(f"Remote command failed: {stderr_text}")

        output = stdout.read().decode('utf-8')
        runs = [line.strip() for line in output.splitlines() if line.strip()]
        return runs

    finally:
        client.close()


def fetch_remote_run_data(server: str, run_name: str) -> dict[str, Any]:
    """
    Fetch run data from a remote server via SSH.

    This reads the meta.json, all metric CSVs, and system.csv from the remote
    server and returns them as a structured dictionary.

    Parameters
    ----------
    server : str
        Server in format "user@host" or "host".
    run_name : str
        Name of the run to fetch.

    Returns
    -------
    dict
        Run data with keys: meta, metrics (dict of metric_name -> csv content),
        system_csv (csv content).

    Raises
    ------
    RuntimeError
        If SSH connection fails or run doesn't exist.
    FileNotFoundError
        If the run is not found on remote server.
    """
    _ensure_paramiko()
    username, hostname = parse_ssh_host(server)

    client = create_ssh_client(username, hostname)
    try:
        run_dir = f"~/.ex/runs/{run_name}"

        # Check if run exists
        stdin, stdout, stderr = client.exec_command(
            f'test -d {run_dir} && test -f {run_dir}/meta.json && echo "exists"'
        )
        exists = stdout.read().decode('utf-8').strip() == "exists"
        if not exists:
            raise FileNotFoundError(
                f"Run '{run_name}' not found on {username}@{hostname}"
            )

        # Read meta.json
        stdin, stdout, stderr = client.exec_command(f'cat {run_dir}/meta.json')
        meta_content = stdout.read().decode('utf-8')
        meta = json.loads(meta_content)

        # List all metric CSV files
        stdin, stdout, stderr = client.exec_command(
            f'cd {run_dir}/metrics 2>/dev/null && find . -name "*.csv" -type f || true'
        )
        metric_files = [
            line.strip().lstrip('./')
            for line in stdout.read().decode('utf-8').splitlines()
            if line.strip()
        ]

        # Read each metric CSV
        metrics = {}
        for metric_file in metric_files:
            stdin, stdout, stderr = client.exec_command(
                f'cat {run_dir}/metrics/{metric_file}'
            )
            csv_content = stdout.read().decode('utf-8')
            # Remove .csv extension and use as metric name
            metric_name = metric_file.replace('.csv', '')
            metrics[metric_name] = csv_content

        # Read system.csv if it exists
        system_csv = None
        stdin, stdout, stderr = client.exec_command(
            f'test -f {run_dir}/system.csv && cat {run_dir}/system.csv || true'
        )
        system_content = stdout.read().decode('utf-8')
        if system_content.strip():
            system_csv = system_content

        return {
            'meta': meta,
            'metrics': metrics,
            'system_csv': system_csv,
        }

    finally:
        client.close()


def stream_remote_run_data(server: str, run_name: str) -> dict:
    """
    Stream run data from remote server and convert to the same format as local sync.

    This is similar to _build_payload in sync.py but reads from SSH.

    Parameters
    ----------
    server : str
        Server in format "user@host" or "host".
    run_name : str
        Name of the run to stream.

    Returns
    -------
    dict
        Payload in the same format as sync._build_payload().
    """
    data = fetch_remote_run_data(server, run_name)
    meta = data['meta']

    payload = {
        "project": meta.get("project", "unknown"),
        "run_name": meta.get("run_name", run_name),
        "config": meta.get("config"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "status": meta.get("status", "completed"),
        "metrics": {},
        "system": [],
    }

    # Parse metric CSVs
    for metric_name, csv_content in data['metrics'].items():
        points = _parse_metric_csv_string(csv_content)
        if points:
            payload["metrics"][metric_name] = points

    # Parse system CSV
    if data['system_csv']:
        payload["system"] = _parse_system_csv_string(data['system_csv'])

    return payload


def _parse_metric_csv_string(csv_content: str) -> list[dict]:
    """Parse a metric CSV string into a list of points."""
    points = []
    lines = csv_content.strip().splitlines()

    for line in lines[1:]:  # Skip header
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


def _parse_system_csv_string(csv_content: str) -> list[dict]:
    """Parse system.csv string into a list of system points."""
    points = []
    lines = csv_content.strip().splitlines()

    for line in lines[1:]:  # Skip header
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
