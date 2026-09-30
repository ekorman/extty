"""S3 storage backend for extty."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extty._logger import _DIM_CYAN, _NEON_CYAN, _NEON_GREEN, _RESET, _supports_color
from extty._logger import log as logger
from extty.chart import Chart
from extty.checkpoints import (
    META_FILE,
    adopt_download,
    checkpoint_dir,
    file_names,
    new_staging_dir,
    read_checkpoint,
    read_meta,
    remote_relpath,
    required_files,
    save_identity,
    write_meta,
)
from extty.confusion import ConfusionMatrix
from extty.image import Image
from extty.storage import (
    ChartRecord,
    Checkpoint,
    ConfusionMatrixRecord,
    ExampleRecord,
    ImageRecord,
    MetaData,
    MetricPoint,
    SystemMetricPoint,
    dedupe_image_records,
    get_extty_home,
    get_run_dir,
    parse_chart_jsonl,
    parse_checkpoints_json,
    parse_confusion_jsonl,
    parse_examples_jsonl,
    parse_images_jsonl,
    parse_metric_csv,
    parse_system_csv,
    sanitize_metric_name,
)


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
    progress = _TransferProgress(total, bar_label)
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


def _upload_with_progress(
    client: Any,
    local_path: Path,
    bucket: str,
    s3_key: str,
    *,
    label: str | None = None,
) -> None:
    """Upload ``local_path`` to S3 with a progress bar."""
    total = local_path.stat().st_size
    bar_label = label or local_path.name
    logger.info(
        "uploading %s -> s3://%s/%s (%s)",
        local_path,
        bucket,
        s3_key,
        _format_bytes(total),
    )
    progress = _TransferProgress(total, bar_label)
    try:
        client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=s3_key,
            Callback=progress,
            Config=_default_transfer_config(),
        )
    except BaseException:
        progress.abort()
        raise
    progress.finish()


class _TransferProgress:
    """boto3 ``Callback`` that renders an in-place transfer progress bar to stderr."""

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
    from boto3.exceptions import Boto3Error
    from botocore.exceptions import BotoCoreError, ClientError

    _S3_ERRORS: tuple[type[Exception], ...] = (BotoCoreError, ClientError, Boto3Error)
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

        Checks environment variables first, then ``<extty home>/s3/config.toml``.

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
        Load S3Config from ``<extty home>/s3/config.toml``.

        Returns
        -------
        S3Config or None
            Configuration if file exists and has a bucket, None otherwise.
        """
        config_path = get_extty_home() / "s3" / "config.toml"
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


def _make_s3_client(config: S3Config) -> Any:
    """Build a boto3 S3 client from the given config."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 storage. Install with: pip install extty[s3]"
        )

    kwargs: dict[str, Any] = {"config": _default_boto_config()}
    if config.region:
        kwargs["region_name"] = config.region
    if config.access_key_id and config.secret_access_key:
        kwargs["aws_access_key_id"] = config.access_key_id
        kwargs["aws_secret_access_key"] = config.secret_access_key
    if config.endpoint_url:
        kwargs["endpoint_url"] = config.endpoint_url

    return boto3.client("s3", **kwargs)


def _run_key_prefix(config: S3Config, project: str, run_name: str) -> str:
    """Return the S3 key prefix (no trailing slash) for a single run."""
    parts = ["runs", project, run_name]
    if config.prefix:
        parts.insert(0, config.prefix)
    return "/".join(parts)


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
    _confusion_buffer: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _chart_buffer: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _image_buffer: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _system_buffer: list[
        tuple[float, float, float, float | None, float | None, float | None]
    ] = field(default_factory=list, repr=False)
    _buffer_count: int = field(default=0, repr=False)
    _last_flush: float = field(default_factory=time.time, repr=False)
    _buffer_max_count: int = 500
    _buffer_max_seconds: float = 30.0
    _consecutive_failures: int = field(default=0, repr=False)
    _max_backoff_seconds: float = 300.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    _cumulative_metric_rows: dict[str, list[tuple[int, float, float]]] = field(
        default_factory=dict, repr=False
    )
    _cumulative_example_rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _cumulative_confusion_rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _cumulative_chart_rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _cumulative_image_rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _cumulative_system_rows: list[
        tuple[float, float, float, float | None, float | None, float | None]
    ] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._init_client()

    def _init_client(self) -> None:
        self._client = _make_s3_client(self.config)

    def _s3_key(self, *parts: str) -> str:
        return "/".join(
            (_run_key_prefix(self.config, self.project, self.run_name), *parts)
        )

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

    def log_confusion_matrix(self, name: str, cm: ConfusionMatrix, step: int) -> None:
        timestamp = time.time()
        record = {
            "step": step,
            "timestamp": timestamp,
            "labels": cm.labels,
            "matrix": cm.matrix,
        }
        with self._lock:
            if name not in self._confusion_buffer:
                self._confusion_buffer[name] = []
            self._confusion_buffer[name].append(record)
            self._buffer_count += 1
            self._maybe_flush()

    def log_chart(self, name: str, chart: Chart, step: int) -> None:
        timestamp = time.time()
        record = {
            "step": step,
            "timestamp": timestamp,
            "x_axis": chart.axis_names[0],
            "y_axis": chart.axis_names[1],
            "points": [[x, y] for x, y in chart.points],
        }
        with self._lock:
            if name not in self._chart_buffer:
                self._chart_buffer[name] = []
            self._chart_buffer[name].append(record)
            self._buffer_count += 1
            self._maybe_flush()

    def log_image(self, name: str, image: Image, step: int) -> None:
        timestamp = time.time()
        safe_name = sanitize_metric_name(name)
        record: dict[str, Any] = {
            "step": step,
            "timestamp": timestamp,
            "file": f"{safe_name}/step_{step}.png",
            "width": image.width,
            "height": image.height,
        }
        if image.caption is not None:
            record["caption"] = image.caption
        with self._lock:
            if name not in self._image_buffer:
                self._image_buffer[name] = []
            self._image_buffer[name].append({"record": record, "png": image.png_bytes})
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
        age = time.time() - self._last_flush
        if self._buffer_count >= self._buffer_max_count:
            reason = f"count ({self._buffer_count} >= {self._buffer_max_count})"
        elif age >= flush_interval:
            reason = f"age ({age:.1f}s >= {flush_interval:.1f}s)"
        else:
            return
        self._flush_unlocked(reason=reason)

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked(reason="manual")

    def _flush_unlocked(self, *, reason: str = "manual") -> None:
        if self._buffer_count == 0:
            return

        logger.debug(
            "S3 flush: %d items (%d metric / %d example / %d confusion / %d chart "
            "/ %d image streams, %d system samples) — trigger: %s",
            self._buffer_count,
            len(self._metric_buffer),
            len(self._example_buffer),
            len(self._confusion_buffer),
            len(self._chart_buffer),
            len(self._image_buffer),
            len(self._system_buffer),
            reason,
        )

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

        failed_confusion: dict[str, list[dict[str, Any]]] = {}
        for name, records in self._confusion_buffer.items():
            try:
                self._upload_confusion(name, records)
            except _S3_ERRORS:
                logger.warning(
                    "Failed to upload confusion matrix '%s' to S3",
                    name,
                    exc_info=True,
                )
                failed_confusion[name] = records
                had_failure = True
        self._confusion_buffer.clear()
        self._confusion_buffer.update(failed_confusion)

        failed_chart: dict[str, list[dict[str, Any]]] = {}
        for name, records in self._chart_buffer.items():
            try:
                self._upload_chart(name, records)
            except _S3_ERRORS:
                logger.warning(
                    "Failed to upload chart '%s' to S3",
                    name,
                    exc_info=True,
                )
                failed_chart[name] = records
                had_failure = True
        self._chart_buffer.clear()
        self._chart_buffer.update(failed_chart)

        failed_images: dict[str, list[dict[str, Any]]] = {}
        for name, items in self._image_buffer.items():
            try:
                self._upload_images(name, items)
            except _S3_ERRORS:
                logger.warning(
                    "Failed to upload images '%s' to S3",
                    name,
                    exc_info=True,
                )
                failed_images[name] = items
                had_failure = True
        self._image_buffer.clear()
        self._image_buffer.update(failed_images)

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
            + sum(len(v) for v in self._confusion_buffer.values())
            + sum(len(v) for v in self._chart_buffer.values())
            + sum(len(v) for v in self._image_buffer.values())
            + len(self._system_buffer)
        )
        self._last_flush = time.time()

    def _put_csv_metric(self, key: str, rows: list[tuple[int, float, float]]) -> None:
        output = io.StringIO()
        output.write("step,timestamp,value\n")
        for step, ts, val in rows:
            output.write(f"{step},{ts:.6f},{val}\n")
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=output.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

    def _put_jsonl(self, key: str, records: list[dict[str, Any]]) -> None:
        output = "\n".join(json.dumps(r) for r in records)
        if output:
            output += "\n"
        self._client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=output.encode("utf-8"),
            ContentType="application/x-ndjson",
        )

    def _put_csv_system(
        self,
        key: str,
        rows: list[
            tuple[float, float, float, float | None, float | None, float | None]
        ],
    ) -> None:
        output = io.StringIO()
        output.write(
            "timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct\n"
        )
        for ts, ram_used, ram_total, gpu_used, gpu_total, gpu_util in rows:
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

    def _upload_metrics(
        self, name: str, values: list[tuple[int, float, float]]
    ) -> None:
        safe_name = sanitize_metric_name(name)
        key = self._s3_key("metrics", f"{safe_name}.csv")
        rows = list(self._cumulative_metric_rows.get(name, [])) + list(values)
        self._put_csv_metric(key, rows)
        self._cumulative_metric_rows[name] = rows

    def _upload_examples(self, name: str, records: list[dict[str, Any]]) -> None:
        safe_name = sanitize_metric_name(name)
        key = self._s3_key("examples", f"{safe_name}.jsonl")
        all_records = list(self._cumulative_example_rows.get(name, [])) + list(records)
        self._put_jsonl(key, all_records)
        self._cumulative_example_rows[name] = all_records

    def _upload_confusion(self, name: str, records: list[dict[str, Any]]) -> None:
        safe_name = sanitize_metric_name(name)
        key = self._s3_key("confusion_matrices", f"{safe_name}.jsonl")
        all_records = list(self._cumulative_confusion_rows.get(name, [])) + list(
            records
        )
        self._put_jsonl(key, all_records)
        self._cumulative_confusion_rows[name] = all_records

    def _upload_chart(self, name: str, records: list[dict[str, Any]]) -> None:
        safe_name = sanitize_metric_name(name)
        key = self._s3_key("charts", f"{safe_name}.jsonl")
        all_records = list(self._cumulative_chart_rows.get(name, [])) + list(records)
        self._put_jsonl(key, all_records)
        self._cumulative_chart_rows[name] = all_records

    def _upload_images(self, name: str, items: list[dict[str, Any]]) -> None:
        """Upload pending PNGs (write-once objects) then rewrite the index JSONL.

        PNG uploads are idempotent (deterministic key per step), so if the
        index rewrite fails the whole batch is retried safely on next flush.
        """
        for item in items:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=self._s3_key("images", item["record"]["file"]),
                Body=item["png"],
                ContentType="image/png",
            )
        safe_name = sanitize_metric_name(name)
        key = self._s3_key("images", f"{safe_name}.jsonl")
        new_records = [item["record"] for item in items]
        all_records = list(self._cumulative_image_rows.get(name, [])) + new_records
        self._put_jsonl(key, all_records)
        self._cumulative_image_rows[name] = all_records

    def _upload_system(
        self,
        values: list[
            tuple[float, float, float, float | None, float | None, float | None]
        ],
    ) -> None:
        key = self._s3_key("system.csv")
        rows = list(self._cumulative_system_rows) + list(values)
        self._put_csv_system(key, rows)
        self._cumulative_system_rows = rows

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

    def upload_checkpoint(
        self, meta: dict[str, Any], sources: Mapping[str, Path]
    ) -> bool:
        """
        Upload a save prepared by :func:`extty.checkpoints.stage_checkpoint`.

        The files go under the save's own prefix, then the step's
        ``meta.json`` is written to point at them, so a failure part way
        leaves S3's copy of the step as it was. Files of the save it replaces
        are removed afterwards.

        S3 errors are non-fatal: they are logged as warnings and reported
        through the return value, so the caller can keep the local copy.

        Parameters
        ----------
        meta : dict[str, Any]
            The save's meta entry.
        sources : Mapping[str, Path]
            Where each of the save's files is, keyed by file name.

        Returns
        -------
        bool
            True if the files and ``meta.json`` were uploaded and the index
            was updated.
        """
        step = meta["step"]
        try:
            for name, source in sources.items():
                _upload_with_progress(
                    self._client,
                    source,
                    self.config.bucket,
                    self._checkpoint_file_key(step, meta, name),
                    label=f"step {step} / {name}",
                )
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=self._s3_key("checkpoints", str(step), META_FILE),
                Body=json.dumps(meta, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except _S3_ERRORS:
            logger.warning(
                "Failed to save checkpoint (step %d) to S3", step, exc_info=True
            )
            return False
        self._remove_superseded_saves(meta)
        return self._update_checkpoints_index(meta)

    def _checkpoint_file_key(self, step: int, meta: dict[str, Any], name: str) -> str:
        return self._s3_key("checkpoints", str(step), remote_relpath(meta, name))

    def _remove_superseded_saves(self, meta: dict[str, Any]) -> None:
        """Delete a step's objects that belong to saves other than ``meta``'s."""
        step_prefix = self._s3_key("checkpoints", str(meta["step"]), "")
        keep = {step_prefix + META_FILE}
        keep_prefix = step_prefix + f"{meta['save_id']}/"
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self.config.bucket, Prefix=step_prefix
            ):
                stale = [
                    {"Key": obj["Key"]}
                    for obj in page.get("Contents", [])
                    if obj["Key"] not in keep and not obj["Key"].startswith(keep_prefix)
                ]
                if stale:
                    self._client.delete_objects(
                        Bucket=self.config.bucket, Delete={"Objects": stale}
                    )
        except _S3_ERRORS:
            logger.warning(
                "Failed to remove superseded files of checkpoint step %d",
                meta["step"],
                exc_info=True,
            )

    def _update_checkpoints_index(self, entry: dict[str, Any]) -> bool:
        """
        Merge ``entry`` into the run's ``checkpoints.json`` in S3.

        A missing or corrupt index starts empty, but one that can't be read
        is left alone: rewriting it with only ``entry`` would drop every other
        step.

        Returns
        -------
        bool
            True if the index was written.
        """
        index_key = self._s3_key("checkpoints.json")
        existing: list[dict[str, Any]] = []
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=index_key)
            existing = json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
            pass
        except json.JSONDecodeError:
            logger.warning("Rebuilding unreadable checkpoints index in S3")
        except _S3_ERRORS:
            logger.warning("Failed to read checkpoints index from S3", exc_info=True)
            return False

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
            return False
        return True

    def load_checkpoint(
        self,
        step: int,
        load_optimizer: bool = True,
        map_location=None,
    ) -> dict[str, Any]:
        """
        Load S3's save of a checkpoint, reusing what is already local.

        A local copy of the same save is used as far as it goes, and any file
        it lacks is added to it. Otherwise the save is downloaded to a staging
        directory and becomes the local copy, unless the local copy is a
        different save, which is left alone. Several processes, such as the
        ranks of a distributed job, can load the same step at once.

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
            If S3 has no checkpoint for the step.
        """
        meta = self._checkpoint_meta(step)
        names = file_names(meta)
        needed = required_files(names, load_optimizer=load_optimizer)
        run_dir = get_run_dir(self.project, self.run_name)
        local_dir = checkpoint_dir(run_dir, step)
        local_meta = read_meta(local_dir)
        same_save = local_meta is not None and save_identity(
            local_meta
        ) == save_identity(meta)
        missing = [
            name for name in needed if not (same_save and (local_dir / name).exists())
        ]

        staging = new_staging_dir(run_dir)
        try:
            self._download_checkpoint_files(step, meta, missing, staging)
            write_meta(staging, meta)
            if adopt_download(staging, local_dir, meta):
                source = local_dir
            else:
                logger.warning(
                    "checkpoint step %d: %s holds a different save than S3; "
                    "loading S3's without replacing it",
                    step,
                    local_dir,
                )
                source = staging
            return read_checkpoint(
                source, names, load_optimizer=load_optimizer, map_location=map_location
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _download_checkpoint_files(
        self, step: int, meta: dict[str, Any], names: list[str], dest: Path
    ) -> None:
        """Download ``names`` from ``meta``'s save into the directory ``dest``."""
        for name in names:
            _download_with_progress(
                self._client,
                self.config.bucket,
                self._checkpoint_file_key(step, meta, name),
                dest / name,
                label=f"step {step} / {name}",
            )

    def _checkpoint_meta(self, step: int) -> dict[str, Any]:
        """
        Fetch S3's meta entry for a checkpoint step.

        The step's ``meta.json`` is read first: it is written last on every
        save, so it always describes the save whose files are in S3. The
        index is only consulted for checkpoints old enough to lack one.

        Raises
        ------
        FileNotFoundError
            If S3 has no checkpoint for the step.
        """
        meta_key = self._s3_key("checkpoints", str(step), META_FILE)
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=meta_key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except self._client.exceptions.NoSuchKey:
            pass
        for entry in self.list_checkpoints():
            if entry.get("step") == step:
                return entry
        raise FileNotFoundError(f"Checkpoint step {step} not found.")

    def delete_checkpoint_optimizer(self, step: int) -> None:
        """
        Delete just the optimizer file of a checkpoint from S3.

        Removes ``optimizer.pt`` from the checkpoint's S3 prefix and
        rewrites both the per-step ``meta.json`` and the per-run
        ``checkpoints.json`` index so neither references the optimizer
        any more. The ``model.pt`` (or legacy ``checkpoint.pt``) is
        left untouched.

        Idempotent: if the checkpoint exists but has no optimizer
        recorded, returns without error and without writing.

        Parameters
        ----------
        step : int
            The training step whose optimizer should be removed.

        Raises
        ------
        FileNotFoundError
            If the checkpoint step does not exist.
        """
        meta = self._checkpoint_meta(step)
        files = meta.get("files", [])

        def is_optimizer(entry: Any) -> bool:
            if isinstance(entry, str):
                return entry == "optimizer.pt"
            if isinstance(entry, dict):
                return entry.get("name") == "optimizer.pt"
            return False

        if not any(is_optimizer(f) for f in files):
            return

        opt_key = self._checkpoint_file_key(step, meta, "optimizer.pt")
        try:
            self._client.delete_object(Bucket=self.config.bucket, Key=opt_key)
        except _S3_ERRORS:
            logger.warning(
                "Failed to delete optimizer for step %d", step, exc_info=True
            )
            return

        meta["files"] = [f for f in files if not is_optimizer(f)]

        meta_key = self._s3_key("checkpoints", str(step), "meta.json")
        try:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=meta_key,
                Body=json.dumps(meta, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except _S3_ERRORS:
            logger.warning(
                "Failed to update meta.json for step %d after optimizer delete",
                step,
                exc_info=True,
            )

        self._update_checkpoints_index(meta)

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

    def find_checkpoint(self, step: int) -> dict[str, Any] | None:
        """
        Look up the checkpoint for ``step`` in S3.

        Parameters
        ----------
        step : int
            The training step to look up.

        Returns
        -------
        dict[str, Any] or None
            The checkpoint's entry from the index or its ``meta.json``, or
            None if S3 has no checkpoint for ``step``.
        """
        try:
            return self._checkpoint_meta(step)
        except FileNotFoundError:
            return None

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


@dataclass
class S3RunReader:
    """
    Read-only view of a run stored in S3.

    Implements :class:`extty.storage.RunStorageReader` so it can be slotted
    into :class:`extty.query.RunData` interchangeably with the local-disk
    :class:`extty.storage.RunStorage`.
    """

    config: S3Config
    project: str
    run_name: str
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            self._client = _make_s3_client(self.config)

    def _s3_key(self, *parts: str) -> str:
        return "/".join(
            (_run_key_prefix(self.config, self.project, self.run_name), *parts)
        )

    def _get_text(self, key: str) -> str | None:
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=key)
        except self._client.exceptions.NoSuchKey:
            return None
        return response["Body"].read().decode("utf-8")

    def _list_stream_names(self, sub_prefix: str, suffix: str) -> list[str]:
        prefix = self._s3_key(sub_prefix) + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        names: list[str] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(suffix):
                    continue
                relative = key[len(prefix) : -len(suffix)]
                if relative:
                    names.append(relative)
        names.sort()
        return names

    def read_meta(self) -> MetaData | None:
        text = self._get_text(self._s3_key("meta.json"))
        if text is None:
            return None
        return MetaData.from_dict(json.loads(text))

    def list_metric_names(self) -> list[str]:
        return self._list_stream_names("metrics", ".csv")

    def read_metric(self, name: str) -> list[MetricPoint]:
        key = self._s3_key("metrics", sanitize_metric_name(name) + ".csv")
        text = self._get_text(key)
        if text is None:
            raise FileNotFoundError(
                f"Metric '{name}' not found at s3://{self.config.bucket}/{key}"
            )
        return parse_metric_csv(text)

    def read_system_metrics(self) -> list[SystemMetricPoint]:
        text = self._get_text(self._s3_key("system.csv"))
        if text is None:
            return []
        return parse_system_csv(text)

    def list_example_names(self) -> list[str]:
        return self._list_stream_names("examples", ".jsonl")

    def read_examples(self, name: str) -> list[ExampleRecord]:
        key = self._s3_key("examples", sanitize_metric_name(name) + ".jsonl")
        text = self._get_text(key)
        if text is None:
            raise FileNotFoundError(
                f"Examples '{name}' not found at s3://{self.config.bucket}/{key}"
            )
        return parse_examples_jsonl(text)

    def list_confusion_matrix_names(self) -> list[str]:
        return self._list_stream_names("confusion_matrices", ".jsonl")

    def read_confusion_matrix(self, name: str) -> list[ConfusionMatrixRecord]:
        key = self._s3_key("confusion_matrices", sanitize_metric_name(name) + ".jsonl")
        text = self._get_text(key)
        if text is None:
            raise FileNotFoundError(
                f"Confusion matrix '{name}' not found at s3://{self.config.bucket}/{key}"
            )
        return parse_confusion_jsonl(text)

    def list_chart_names(self) -> list[str]:
        return self._list_stream_names("charts", ".jsonl")

    def read_chart(self, name: str) -> list[ChartRecord]:
        key = self._s3_key("charts", sanitize_metric_name(name) + ".jsonl")
        text = self._get_text(key)
        if text is None:
            raise FileNotFoundError(
                f"Chart '{name}' not found at s3://{self.config.bucket}/{key}"
            )
        return parse_chart_jsonl(text)

    def list_image_names(self) -> list[str]:
        return self._list_stream_names("images", ".jsonl")

    def read_images(self, name: str) -> list[ImageRecord]:
        key = self._s3_key("images", sanitize_metric_name(name) + ".jsonl")
        text = self._get_text(key)
        if text is None:
            raise FileNotFoundError(
                f"Images '{name}' not found at s3://{self.config.bucket}/{key}"
            )
        return dedupe_image_records(parse_images_jsonl(text))

    def read_image_bytes(self, file: str) -> bytes:
        key = self._s3_key("images", file)
        try:
            response = self._client.get_object(Bucket=self.config.bucket, Key=key)
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(
                f"Image file '{file}' not found at s3://{self.config.bucket}/{key}"
            ) from None
        return response["Body"].read()

    def read_checkpoints(self) -> list[Checkpoint]:
        text = self._get_text(self._s3_key("checkpoints.json"))
        if text is None:
            return []
        return parse_checkpoints_json(text)
