"""S3 storage backend for extty."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extty._logger import (
    _DIM_CYAN,
    _NEON_CYAN,
    _NEON_GREEN,
    _RESET,
    _supports_color,
)

logger = logging.getLogger(__name__)


def _format_bytes(n: float) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PiB"


def _default_boto_config() -> Any:
    """Botocore client config tuned for resilient checkpoint/artifact transfers."""
    from botocore.config import Config

    return Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=120,
        max_pool_connections=20,
    )


def _default_transfer_config() -> Any:
    """boto3 ``TransferConfig`` for multi-part downloads with chunk retries."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_concurrency=8,
        num_download_attempts=10,
        use_threads=True,
    )


def _download_with_progress(
    client: Any,
    bucket: str,
    s3_key: str,
    local_path: Path,
    *,
    label: str | None = None,
) -> None:
    """Download an S3 object to ``local_path`` with a progress bar.

    Streams to ``<local_path>.part`` and renames atomically on success so
    a partial download from a previous failed run isn't mistaken for a
    valid cache entry.
    """
    head = client.head_object(Bucket=bucket, Key=s3_key)
    total = int(head.get("ContentLength", 0))
    bar_label = label or local_path.name
    logger.info(
        "downloading s3://%s/%s -> %s (%s)",
        bucket,
        s3_key,
        local_path,
        _format_bytes(total),
    )
    tmp_path = local_path.with_name(local_path.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    progress = _DownloadProgress(total, bar_label)
    try:
        client.download_file(
            Bucket=bucket,
            Key=s3_key,
            Filename=str(tmp_path),
            Callback=progress,
            Config=_default_transfer_config(),
        )
    except BaseException:
        progress.abort()
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    progress.finish()
    tmp_path.replace(local_path)


class _DownloadProgress:
    """boto3 ``Callback`` that renders an in-place progress bar to stderr."""

    def __init__(self, total_bytes: int, label: str, *, width: int = 30) -> None:
        self._total = max(total_bytes, 1)
        self._seen = 0
        self._label = label
        self._width = width
        self._stream = sys.stderr
        self._enabled = _supports_color(self._stream)
        self._lock = threading.Lock()
        self._last_render = 0.0

    def __call__(self, bytes_transferred: int) -> None:
        with self._lock:
            self._seen += bytes_transferred
            if not self._enabled:
                return
            now = time.time()
            if now - self._last_render < 0.1 and self._seen < self._total:
                return
            self._last_render = now
            self._render()

    def _render(self, end: str = "") -> None:
        frac = min(self._seen / self._total, 1.0)
        filled = int(self._width * frac)
        bar = "█" * filled + "░" * (self._width - filled)
        stats = (
            f"{frac * 100:5.1f}% "
            f"{_format_bytes(self._seen)} / {_format_bytes(self._total)}"
        )
        prefix_visible_len = self._width + len(stats) + 4
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 80
        budget = max(term_width - prefix_visible_len - 1, 0)
        label = self._label
        if len(label) > budget:
            if budget <= 1:
                label = ""
            else:
                label = "…" + label[-(budget - 1) :]
        line = (
            f"{_DIM_CYAN}[{_RESET}"
            f"{_NEON_GREEN}{bar}{_RESET}"
            f"{_DIM_CYAN}]{_RESET} "
            f"{_NEON_CYAN}{frac * 100:5.1f}%{_RESET} "
            f"{_format_bytes(self._seen)} / {_format_bytes(self._total)}"
        )
        if label:
            line += f" {_DIM_CYAN}{label}{_RESET}"
        self._stream.write(f"\r\033[2K{line}{end}")
        self._stream.flush()

    def finish(self) -> None:
        if self._enabled:
            with self._lock:
                self._render(end="\n")

    def abort(self) -> None:
        if self._enabled:
            with self._lock:
                self._stream.write("\n")
                self._stream.flush()


try:
    from botocore.exceptions import BotoCoreError, ClientError

    _S3_ERRORS: tuple[type[Exception], ...] = (BotoCoreError, ClientError)
except ImportError:
    _S3_ERRORS = ()


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
    _consecutive_failures: int = field(default=0, repr=False)
    _max_backoff_seconds: float = 300.0
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

        kwargs: dict[str, Any] = {"config": _default_boto_config()}
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
        max_doublings = 4
        flush_interval = self._buffer_max_seconds * (
            2 ** min(self._consecutive_failures, max_doublings)
        )
        should_flush = (
            self._buffer_count >= self._buffer_max_count
            or (time.time() - self._last_flush) >= flush_interval
        )
        if should_flush:
            self._flush_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if self._buffer_count == 0:
            return

        had_failure = False

        failed_metrics: dict[str, list[tuple[int, float, float]]] = {}
        for name, values in self._metric_buffer.items():
            try:
                self._upload_metrics(name, values)
            except _S3_ERRORS:
                logger.warning(
                    "Failed to upload metrics '%s' to S3", name, exc_info=True
                )
                failed_metrics[name] = values
                had_failure = True
        self._metric_buffer.clear()
        self._metric_buffer.update(failed_metrics)

        failed_examples: dict[str, list[dict[str, Any]]] = {}
        for name, records in self._example_buffer.items():
            try:
                self._upload_examples(name, records)
            except _S3_ERRORS:
                logger.warning(
                    "Failed to upload examples '%s' to S3", name, exc_info=True
                )
                failed_examples[name] = records
                had_failure = True
        self._example_buffer.clear()
        self._example_buffer.update(failed_examples)

        if self._system_buffer:
            try:
                self._upload_system(self._system_buffer)
                self._system_buffer.clear()
            except _S3_ERRORS:
                logger.warning("Failed to upload system metrics to S3", exc_info=True)
                had_failure = True

        if had_failure:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        self._buffer_count = (
            sum(len(v) for v in self._metric_buffer.values())
            + sum(len(v) for v in self._example_buffer.values())
            + len(self._system_buffer)
        )
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
        try:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=json.dumps(meta_dict, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except _S3_ERRORS:
            logger.warning("Failed to write run metadata to S3", exc_info=True)

    def save_checkpoint(
        self,
        step: int,
        path: str | None = None,
        state_dict: Any = None,
        optimizer_state_dict: Any = None,
    ) -> None:
        """
        Save a checkpoint to S3.

        S3 upload errors are non-fatal: failures are logged as warnings
        and the method returns without raising. The checkpoint may not
        be persisted remotely in that case.

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

        try:
            if path is not None:
                file_size = os.path.getsize(path)
                checkpoint_key = self._s3_key("checkpoints", str(step), "checkpoint.pt")
                self._client.upload_file(
                    Filename=path,
                    Bucket=self.config.bucket,
                    Key=checkpoint_key,
                    ExtraArgs={"ContentType": "application/octet-stream"},
                )

                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                meta_entry: dict[str, Any] = {
                    "step": step,
                    "timestamp": timestamp,
                    "files": [{"name": "checkpoint.pt", "size_bytes": file_size}],
                }

                meta_key = self._s3_key("checkpoints", str(step), "meta.json")
                self._client.put_object(
                    Bucket=self.config.bucket,
                    Key=meta_key,
                    Body=json.dumps(meta_entry, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )
                self._update_checkpoints_index(meta_entry)
            else:
                import torch

                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                files_meta: list[dict[str, Any]] = []
                tmp_files: list[str] = []

                try:
                    model_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                    tmp_files.append(model_tmp.name)
                    torch.save(state_dict, model_tmp.name)
                    model_tmp.close()

                    model_size = os.path.getsize(model_tmp.name)
                    self._client.upload_file(
                        Filename=model_tmp.name,
                        Bucket=self.config.bucket,
                        Key=self._s3_key("checkpoints", str(step), "model.pt"),
                        ExtraArgs={"ContentType": "application/octet-stream"},
                    )
                    files_meta.append({"name": "model.pt", "size_bytes": model_size})

                    if optimizer_state_dict is not None:
                        opt_tmp = tempfile.NamedTemporaryFile(
                            suffix=".pt", delete=False
                        )
                        tmp_files.append(opt_tmp.name)
                        torch.save(optimizer_state_dict, opt_tmp.name)
                        opt_tmp.close()

                        opt_size = os.path.getsize(opt_tmp.name)
                        self._client.upload_file(
                            Filename=opt_tmp.name,
                            Bucket=self.config.bucket,
                            Key=self._s3_key("checkpoints", str(step), "optimizer.pt"),
                            ExtraArgs={"ContentType": "application/octet-stream"},
                        )
                        files_meta.append(
                            {"name": "optimizer.pt", "size_bytes": opt_size}
                        )

                    meta_entry = {
                        "step": step,
                        "timestamp": timestamp,
                        "files": files_meta,
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
                    for f in tmp_files:
                        os.unlink(f)
        except _S3_ERRORS:
            logger.warning(
                "Failed to save checkpoint (step %d) to S3", step, exc_info=True
            )

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

        try:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=index_key,
                Body=json.dumps(existing, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except _S3_ERRORS:
            logger.warning("Failed to update checkpoints index in S3", exc_info=True)

    def load_checkpoint(
        self,
        step: int,
        load_optimizer: bool = True,
        map_location=None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint, downloading from S3 if not cached locally.

        Handles both old format (single ``checkpoint.pt`` containing
        ``model_state_dict`` and ``optimizer_state_dict`` keys) and new
        format (separate ``model.pt`` / ``optimizer.pt`` files).

        Parameters
        ----------
        step : int
            The training step to load.
        load_optimizer : bool, default True
            Whether to include the optimizer state in the result.

        Returns
        -------
        dict[str, Any]
            Always contains ``"model_state_dict"``.
            Contains ``"optimizer_state_dict"`` when available and
            *load_optimizer* is True.

        Raises
        ------
        FileNotFoundError
            If the checkpoint step does not exist in the index.
        """
        from extty.storage import get_runs_dir

        meta = self._checkpoint_meta(step)

        import torch

        map_location = map_location or torch.device("cpu")

        project_dir = self.project if self.project else "_default"
        local_dir = (
            get_runs_dir() / project_dir / self.run_name / "checkpoints" / str(step)
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        files_raw = meta.get("files", [])

        file_names: list[str] = []
        for entry in files_raw:
            if isinstance(entry, str):
                file_names.append(entry)
            elif isinstance(entry, dict):
                file_names.append(entry["name"])

        if not file_names:
            file_names = ["checkpoint.pt"]

        for fname in file_names:
            if not load_optimizer and fname == "optimizer.pt":
                continue
            local_path = local_dir / fname
            if local_path.exists():
                logger.info(
                    "checkpoint step %d: using cached %s (%s)",
                    step,
                    fname,
                    _format_bytes(local_path.stat().st_size),
                )
                continue
            s3_key = self._s3_key("checkpoints", str(step), fname)
            _download_with_progress(
                self._client,
                self.config.bucket,
                s3_key,
                local_path,
                label=f"step {step} / {fname}",
            )

        is_legacy = file_names == ["checkpoint.pt"]
        if is_legacy:
            data = torch.load(
                local_dir / "checkpoint.pt",
                weights_only=False,
                map_location=map_location,
            )
            result: dict[str, Any] = {
                "model_state_dict": data.get("model_state_dict", data)
            }
            if load_optimizer and "optimizer_state_dict" in data:
                result["optimizer_state_dict"] = data["optimizer_state_dict"]
            return result

        result = {
            "model_state_dict": torch.load(
                local_dir / "model.pt", weights_only=False, map_location=map_location
            )
        }
        if (
            load_optimizer
            and "optimizer.pt" in file_names
            and (local_dir / "optimizer.pt").exists()
        ):
            result["optimizer_state_dict"] = torch.load(
                local_dir / "optimizer.pt",
                weights_only=False,
                map_location=map_location,
            )
        return result

    def _checkpoint_meta(self, step: int) -> dict[str, Any]:
        """Fetch the meta entry for a given checkpoint step.

        Falls back to fetching the per-step meta.json directly if the
        checkpoint index is missing or outdated (e.g. due to a failed
        index write).
        """
        for entry in self.list_checkpoints():
            if entry.get("step") == step:
                return entry

        meta_key = self._s3_key("checkpoints", str(step), "meta.json")
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=meta_key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Checkpoint step {step} not found.")

    def delete_checkpoint(self, step: int) -> None:
        """
        Delete a checkpoint from S3.

        Removes all objects under the checkpoint's S3 prefix and updates
        the checkpoints index.

        Parameters
        ----------
        step : int
            The training step to delete.
        """
        prefix = self._s3_key("checkpoints", str(step), "")
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(
                    Bucket=self.config.bucket, Delete={"Objects": objects}
                )

        index_key = self._s3_key("checkpoints.json")
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=index_key)
            existing = json.loads(response["Body"].read().decode("utf-8"))
            updated = [cp for cp in existing if cp.get("step") != step]
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=index_key,
                Body=json.dumps(updated, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception:
            pass

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
