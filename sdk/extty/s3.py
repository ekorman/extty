"""S3 storage backend for extty."""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class S3Config:
    """Configuration for S3 storage."""

    bucket: str
    prefix: str = ""
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    endpoint_url: str | None = None

    @classmethod
    def load(cls) -> S3Config | None:
        """
        Create S3Config from environment variables first then falls back to config file.

        Checks environment variables first, then ~/.extty/s3/config.toml.

        Returns
        -------
        S3Config or None
            Configuration if found, None otherwise.
        """
        bucket = os.environ.get("EXTTY_S3_BUCKET")
        if bucket:
            return cls(
                bucket=bucket,
                prefix=os.environ.get("EXTTY_S3_PREFIX", ""),
                region=os.environ.get("EXTTY_S3_REGION"),
                access_key_id=os.environ.get("EXTTY_S3_ACCESS_KEY_ID"),
                secret_access_key=os.environ.get("EXTTY_S3_SECRET_ACCESS_KEY"),
                endpoint_url=os.environ.get("EXTTY_S3_ENDPOINT_URL"),
            )

        return cls.from_file()

    @classmethod
    def from_file(cls) -> S3Config | None:
        """
        Load S3Config from ~/.extty/s3/config.toml.

        Returns
        -------
        S3Config or None
            Configuration if file exists and has a bucket, None otherwise.
        """
        from pathlib import Path

        config_path = Path.home() / ".extty" / "s3" / "config.toml"
        if not config_path.exists():
            return None

        try:
            content = config_path.read_text()
            config = _parse_toml(content)
            bucket = config.get("bucket")
            if not bucket:
                return None
            return cls(
                bucket=bucket,
                prefix=config.get("prefix") or "",
                region=config.get("region"),
                access_key_id=config.get("access_key_id"),
                secret_access_key=config.get("secret_access_key"),
                endpoint_url=config.get("endpoint_url"),
            )
        except Exception:
            return None


def _parse_toml(content: str) -> dict[str, str | None]:
    """Simple TOML parser for flat key-value config."""
    result: dict[str, str | None] = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            result[key] = value if value else None
    return result


@dataclass
class S3Storage:
    """
    S3 storage backend that implements the StorageSink protocol.

    Buffers data locally and periodically uploads to S3.
    """

    config: S3Config
    project: str
    run_name: str
    _client: Any = field(default=None, repr=False)
    _metric_buffer: dict[str, list[tuple[int, float, float]]] = field(
        default_factory=dict, repr=False
    )
    _example_buffer: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _system_buffer: list[
        tuple[float, float, float, float | None, float | None, float | None]
    ] = field(default_factory=list, repr=False)
    _buffer_count: int = field(default=0, repr=False)
    _last_flush: float = field(default_factory=time.time, repr=False)
    _buffer_max_count: int = 100
    _buffer_max_seconds: float = 30.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._init_client()

    def _init_client(self) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install extty[s3]"
            )

        kwargs: dict[str, Any] = {}
        if self.config.region:
            kwargs["region_name"] = self.config.region
        if self.config.access_key_id and self.config.secret_access_key:
            kwargs["aws_access_key_id"] = self.config.access_key_id
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url

        self._client = boto3.client("s3", **kwargs)

    def _s3_key(self, *parts: str) -> str:
        path_parts = ["runs", self.project, self.run_name, *parts]
        if self.config.prefix:
            path_parts.insert(0, self.config.prefix)
        return "/".join(path_parts)

    def log_metric(self, name: str, value: float, step: int) -> None:
        timestamp = time.time()
        with self._lock:
            if name not in self._metric_buffer:
                self._metric_buffer[name] = []
            self._metric_buffer[name].append((step, timestamp, value))
            self._buffer_count += 1
            self._maybe_flush()

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        timestamp = time.time()
        record = {"step": step, "timestamp": timestamp, "data": data}
        with self._lock:
            if name not in self._example_buffer:
                self._example_buffer[name] = []
            self._example_buffer[name].append(record)
            self._buffer_count += 1
            self._maybe_flush()

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        timestamp = time.time()
        with self._lock:
            self._system_buffer.append(
                (
                    timestamp,
                    ram_used_gb,
                    ram_total_gb,
                    gpu_mem_used_gb,
                    gpu_mem_total_gb,
                    gpu_util_pct,
                )
            )
            self._buffer_count += 1
            self._maybe_flush()

    def _maybe_flush(self) -> None:
        should_flush = (
            self._buffer_count >= self._buffer_max_count
            or (time.time() - self._last_flush) >= self._buffer_max_seconds
        )
        if should_flush:
            self._flush_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if self._buffer_count == 0:
            return

        for name, values in self._metric_buffer.items():
            self._upload_metrics(name, values)
        self._metric_buffer.clear()

        for name, records in self._example_buffer.items():
            self._upload_examples(name, records)
        self._example_buffer.clear()

        if self._system_buffer:
            self._upload_system(self._system_buffer)
            self._system_buffer.clear()

        self._buffer_count = 0
        self._last_flush = time.time()

    def _upload_metrics(
        self, name: str, values: list[tuple[int, float, float]]
    ) -> None:
        safe_name = name.replace("/", "_")
        key = self._s3_key("metrics", f"{safe_name}.csv")

        existing_data: set[tuple[int, float]] = set()
        existing_rows: list[tuple[int, float, float]] = []

        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                step = int(row["step"])
                ts = float(row["timestamp"])
                val = float(row["value"])
                existing_data.add((step, ts))
                existing_rows.append((step, ts, val))
        except self._client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        for step, ts, val in values:
            if (step, ts) not in existing_data:
                existing_data.add((step, ts))
                existing_rows.append((step, ts, val))

        existing_rows.sort(key=lambda x: (x[0], x[1]))

        output = io.StringIO()
        output.write("step,timestamp,value\n")
        for step, ts, val in existing_rows:
            output.write(f"{step},{ts:.6f},{val}\n")

        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=output.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

    def _upload_examples(self, name: str, records: list[dict[str, Any]]) -> None:
        safe_name = name.replace("/", "_")
        key = self._s3_key("examples", f"{safe_name}.jsonl")

        existing_data: set[tuple[int, float]] = set()
        existing_records: list[dict[str, Any]] = []

        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            for line in content.strip().split("\n"):
                if line:
                    record = json.loads(line)
                    existing_data.add((record["step"], record["timestamp"]))
                    existing_records.append(record)
        except self._client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        for record in records:
            key_tuple = (record["step"], record["timestamp"])
            if key_tuple not in existing_data:
                existing_data.add(key_tuple)
                existing_records.append(record)

        existing_records.sort(key=lambda x: (x["step"], x["timestamp"]))

        output = "\n".join(json.dumps(r) for r in existing_records)
        if output:
            output += "\n"

        key = self._s3_key("examples", f"{safe_name}.jsonl")
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=output.encode("utf-8"),
            ContentType="application/x-ndjson",
        )

    def _upload_system(
        self,
        values: list[
            tuple[float, float, float, float | None, float | None, float | None]
        ],
    ) -> None:
        key = self._s3_key("system.csv")

        existing_data: set[float] = set()
        existing_rows: list[
            tuple[float, float, float, float | None, float | None, float | None]
        ] = []

        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                ts = float(row["timestamp"])
                existing_data.add(ts)
                existing_rows.append(
                    (
                        ts,
                        float(row["ram_used_gb"]),
                        float(row["ram_total_gb"]),
                        float(row["gpu_mem_used_gb"])
                        if row["gpu_mem_used_gb"]
                        else None,
                        float(row["gpu_mem_total_gb"])
                        if row["gpu_mem_total_gb"]
                        else None,
                        float(row["gpu_util_pct"]) if row["gpu_util_pct"] else None,
                    )
                )
        except self._client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        for row in values:
            if row[0] not in existing_data:
                existing_data.add(row[0])
                existing_rows.append(row)

        existing_rows.sort(key=lambda x: x[0])

        output = io.StringIO()
        output.write(
            "timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct\n"
        )
        for ts, ram_used, ram_total, gpu_used, gpu_total, gpu_util in existing_rows:
            gpu_used_str = "" if gpu_used is None else f"{gpu_used:.2f}"
            gpu_total_str = "" if gpu_total is None else f"{gpu_total:.2f}"
            gpu_util_str = "" if gpu_util is None else f"{gpu_util:.1f}"
            output.write(
                f"{ts:.6f},{ram_used:.2f},{ram_total:.2f},{gpu_used_str},{gpu_total_str},{gpu_util_str}\n"
            )

        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=output.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

    def write_meta(self, meta_dict: dict[str, Any]) -> None:
        key = self._s3_key("meta.json")
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=json.dumps(meta_dict, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def save_checkpoint(
        self,
        step: int,
        path: str | None = None,
        state_dict: Any = None,
        optimizer_state_dict: Any = None,
    ) -> None:
        """
        Save a checkpoint to S3.

        Parameters
        ----------
        step : int
            The training step for this checkpoint.
        path : str or None
            Path to a local file to upload directly.
        state_dict : Any or None
            Model state dict to serialize with torch.save.
        optimizer_state_dict : Any or None
            Optimizer state dict to include when using state_dict.

        Raises
        ------
        ValueError
            If neither or both of path and state_dict are provided.
        """
        if (path is None) == (state_dict is None):
            raise ValueError("Exactly one of `path` or `state_dict` must be provided.")

        import tempfile

        if path is not None:
            local_path = path
        else:
            import torch

            save_obj: dict[str, Any] = {"model_state_dict": state_dict}
            if optimizer_state_dict is not None:
                save_obj["optimizer_state_dict"] = optimizer_state_dict
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            try:
                torch.save(save_obj, tmp.name)
                tmp.close()
                local_path = tmp.name
            except Exception:
                tmp.close()
                os.unlink(tmp.name)
                raise

        try:
            file_size = os.path.getsize(local_path)
            checkpoint_key = self._s3_key("checkpoints", str(step), "checkpoint.pt")
            with open(local_path, "rb") as f:
                self._client.put_object(
                    Bucket=self.config.bucket,
                    Key=checkpoint_key,
                    Body=f.read(),
                    ContentType="application/octet-stream",
                )

            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            meta_entry = {
                "step": step,
                "timestamp": timestamp,
                "size_bytes": file_size,
                "files": ["checkpoint.pt"],
            }

            meta_key = self._s3_key("checkpoints", str(step), "meta.json")
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=meta_key,
                Body=json.dumps(meta_entry, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            self._update_checkpoints_index(meta_entry)
        finally:
            if state_dict is not None:
                os.unlink(local_path)

    def _update_checkpoints_index(self, entry: dict[str, Any]) -> None:
        index_key = self._s3_key("checkpoints.json")
        existing: list[dict[str, Any]] = []
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=index_key)
            content = response["Body"].read().decode("utf-8")
            existing = json.loads(content)
        except self._client.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

        existing_steps = {e["step"] for e in existing}
        if entry["step"] not in existing_steps:
            existing.append(entry)
        else:
            existing = [entry if e["step"] == entry["step"] else e for e in existing]

        existing.sort(key=lambda e: e["step"])

        self._client.put_object(
            Bucket=self.config.bucket,
            Key=index_key,
            Body=json.dumps(existing, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        List all checkpoints for this run.

        Returns
        -------
        list[dict]
            Parsed contents of checkpoints.json.
        """
        index_key = self._s3_key("checkpoints.json")
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=index_key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except self._client.exceptions.NoSuchKey:
            return []
        except Exception:
            return []

    def close(self) -> None:
        self.flush()
