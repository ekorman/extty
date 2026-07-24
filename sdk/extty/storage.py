"""File I/O, CSV writing, and path management for extty."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from extty.chart import Chart
from extty.confusion import ConfusionMatrix


def get_runs_dir() -> Path:
    """Get the default runs directory (~/.ex/runs/)."""
    return Path.home() / ".extty" / "runs"


def get_artifacts_dir() -> Path:
    """Get the default artifacts directory (~/.extty/artifacts/)."""
    return Path.home() / ".extty" / "artifacts"


@dataclass(frozen=True)
class MetricPoint:
    """A single metric data point."""

    step: int
    timestamp: float
    value: float


@dataclass(frozen=True)
class SystemMetricPoint:
    """A single system metric sample."""

    timestamp: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_mem_used_gb: float | None
    gpu_mem_total_gb: float | None
    gpu_util_pct: float | None


@dataclass(frozen=True)
class ExampleRecord:
    """A single logged example record."""

    step: int
    timestamp: float
    data: dict[str, Any]


@dataclass(frozen=True)
class ConfusionMatrixRecord:
    """A single logged confusion matrix record."""

    step: int
    timestamp: float
    labels: list[str]
    matrix: list[list[int]]


@dataclass(frozen=True)
class ChartRecord:
    """A single logged chart record."""

    step: int
    timestamp: float
    x_axis: str
    y_axis: str
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class CheckpointFile:
    """A single file belonging to a checkpoint."""

    name: str
    size_bytes: int


@dataclass(frozen=True)
class Checkpoint:
    """A single saved checkpoint, as recorded in ``checkpoints.json``."""

    step: int
    timestamp: str
    files: list[CheckpointFile]


def sanitize_metric_name(name: str) -> str:
    """
    Sanitize a metric name for use as a file path.

    Preserves forward slashes as directory separators.
    Sanitizes each path component individually.
    """
    parts = name.split("/")
    sanitized_parts = [re.sub(r"[^\w\-.@]", "_", part) for part in parts]
    return "/".join(sanitized_parts)


def generate_random_name() -> str:
    """Generate a unique run name with timestamp and random suffix."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = os.urandom(2).hex()
    return f"{timestamp}_{suffix}"


def parse_metric_csv(text: str) -> list[MetricPoint]:
    """Parse a metric CSV (``step,timestamp,value``) into MetricPoint records."""
    points: list[MetricPoint] = []
    lines = text.splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        points.append(
            MetricPoint(
                step=int(parts[0]),
                timestamp=float(parts[1]),
                value=float(parts[2]),
            )
        )
    return points


def parse_system_csv(text: str) -> list[SystemMetricPoint]:
    """Parse a system.csv body into SystemMetricPoint records."""
    points: list[SystemMetricPoint] = []
    lines = text.splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        points.append(
            SystemMetricPoint(
                timestamp=float(parts[0]),
                ram_used_gb=float(parts[1]),
                ram_total_gb=float(parts[2]),
                gpu_mem_used_gb=float(parts[3]) if parts[3] else None,
                gpu_mem_total_gb=float(parts[4]) if parts[4] else None,
                gpu_util_pct=float(parts[5]) if parts[5] else None,
            )
        )
    return points


def parse_examples_jsonl(text: str) -> list[ExampleRecord]:
    """Parse an examples JSONL body into ExampleRecord values."""
    records: list[ExampleRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append(
            ExampleRecord(
                step=obj["step"],
                timestamp=obj["timestamp"],
                data=obj["data"],
            )
        )
    return records


def parse_confusion_jsonl(text: str) -> list[ConfusionMatrixRecord]:
    """Parse a confusion-matrix JSONL body into ConfusionMatrixRecord values."""
    records: list[ConfusionMatrixRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append(
            ConfusionMatrixRecord(
                step=obj["step"],
                timestamp=obj["timestamp"],
                labels=obj["labels"],
                matrix=obj["matrix"],
            )
        )
    return records


def parse_chart_jsonl(text: str) -> list[ChartRecord]:
    """Parse a chart JSONL body into ChartRecord values."""
    records: list[ChartRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append(
            ChartRecord(
                step=obj["step"],
                timestamp=obj["timestamp"],
                x_axis=obj["x_axis"],
                y_axis=obj["y_axis"],
                points=[(float(x), float(y)) for x, y in obj["points"]],
            )
        )
    return records


def parse_checkpoints_json(text: str) -> list[Checkpoint]:
    """Parse a ``checkpoints.json`` body into Checkpoint values, sorted by step."""
    entries = json.loads(text)
    checkpoints = [
        Checkpoint(
            step=entry["step"],
            timestamp=entry["timestamp"],
            files=[
                CheckpointFile(name=f["name"], size_bytes=f["size_bytes"])
                for f in entry.get("files", [])
            ],
        )
        for entry in entries
    ]
    checkpoints.sort(key=lambda c: c.step)
    return checkpoints


class RunStorageReader(Protocol):
    """Read-only surface needed by :class:`extty.query.RunData`.

    Both the local-filesystem :class:`RunStorage` and the S3-backed
    ``S3RunReader`` implement this so ``RunData`` doesn't need to care
    where its data lives.
    """

    def read_meta(self) -> MetaData | None: ...
    def list_metric_names(self) -> list[str]: ...
    def read_metric(self, name: str) -> list[MetricPoint]: ...
    def read_system_metrics(self) -> list[SystemMetricPoint]: ...
    def list_example_names(self) -> list[str]: ...
    def read_examples(self, name: str) -> list[ExampleRecord]: ...
    def list_confusion_matrix_names(self) -> list[str]: ...
    def read_confusion_matrix(self, name: str) -> list[ConfusionMatrixRecord]: ...
    def list_chart_names(self) -> list[str]: ...
    def read_chart(self, name: str) -> list[ChartRecord]: ...
    def read_checkpoints(self) -> list[Checkpoint]: ...


@dataclass
class MetaData:
    """Metadata for a run."""

    project: str
    run_name: str
    config: dict[str, Any]
    started_at: str
    finished_at: str | None = None
    status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "run_name": self.run_name,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaData:
        return cls(
            project=data["project"],
            run_name=data["run_name"],
            config=data.get("config", {}),
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
            status=data.get("status", "running"),
        )


@dataclass
class RunStorage:
    """Handles all file I/O for a single run.

    Not thread-safe: ``log_metric``, ``log_example``, ``log_system``, ``flush``,
    and ``close`` mutate internal buffers without locking. In production these
    are driven by a single :class:`extty.async_sink.AsyncSink` worker thread.
    """

    run_dir: Path
    _readonly: bool = field(default=False, repr=False)
    _metric_files: dict[str, Any] = field(default_factory=dict, repr=False)
    _system_file: Any = field(default=None, repr=False)
    _buffer: list[tuple[str, int, float, float]] = field(
        default_factory=list, repr=False
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
    _buffer_size: int = 200
    _last_flush: float = field(default_factory=time.time, repr=False)
    _flush_interval: float = 1.0

    def __post_init__(self) -> None:
        if not self._readonly:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "metrics").mkdir(exist_ok=True)
            (self.run_dir / "examples").mkdir(exist_ok=True)
            (self.run_dir / "confusion_matrices").mkdir(exist_ok=True)
            (self.run_dir / "charts").mkdir(exist_ok=True)

    @classmethod
    def open_readonly(cls, run_dir: Path) -> RunStorage:
        """Open an existing run directory for reading without creating directories."""
        return cls(run_dir=run_dir, _readonly=True)

    def write_meta(self, meta: MetaData) -> None:
        """Write or update the meta.json file."""
        meta_path = self.run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    def read_meta(self) -> MetaData | None:
        """Read the meta.json file if it exists."""
        meta_path = self.run_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return MetaData.from_dict(json.load(f))

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Buffer a metric value for later writing."""
        timestamp = time.time()
        self._buffer.append((name, step, timestamp, value))
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Flush if the combined buffer is full or the flush interval has elapsed."""
        total = (
            len(self._buffer)
            + sum(len(v) for v in self._example_buffer.values())
            + sum(len(v) for v in self._confusion_buffer.values())
            + sum(len(v) for v in self._chart_buffer.values())
        )
        if total == 0:
            return
        should_flush = (
            total >= self._buffer_size
            or (time.time() - self._last_flush) >= self._flush_interval
        )
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered metrics and examples to disk."""
        if self._buffer:
            metrics_by_name: dict[str, list[tuple[int, float, float]]] = {}
            for name, step, timestamp, value in self._buffer:
                if name not in metrics_by_name:
                    metrics_by_name[name] = []
                metrics_by_name[name].append((step, timestamp, value))

            for name, values in metrics_by_name.items():
                self._write_metric_batch(name, values)

            self._buffer.clear()

        if self._example_buffer:
            for name, records in self._example_buffer.items():
                self._write_example_batch(name, records)
            self._example_buffer.clear()

        if self._confusion_buffer:
            for name, records in self._confusion_buffer.items():
                self._write_confusion_batch(name, records)
            self._confusion_buffer.clear()

        if self._chart_buffer:
            for name, records in self._chart_buffer.items():
                self._write_chart_batch(name, records)
            self._chart_buffer.clear()

        self._last_flush = time.time()

    def _write_metric_batch(
        self, name: str, values: list[tuple[int, float, float]]
    ) -> None:
        """Write a batch of metric values to the CSV file."""
        relative_path = sanitize_metric_name(name) + ".csv"
        filepath = self.run_dir / "metrics" / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        file_exists = filepath.exists()
        with open(filepath, "a") as f:
            if not file_exists:
                f.write("step,timestamp,value\n")
            for step, timestamp, value in values:
                f.write(f"{step},{timestamp:.6f},{value}\n")

    def _write_example_batch(self, name: str, records: list[dict[str, Any]]) -> None:
        """Write a batch of example records to the JSONL file."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "examples" / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def _write_confusion_batch(self, name: str, records: list[dict[str, Any]]) -> None:
        """Write a batch of confusion matrix records to the JSONL file."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "confusion_matrices" / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def _write_chart_batch(self, name: str, records: list[dict[str, Any]]) -> None:
        """Write a batch of chart records to the JSONL file."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "charts" / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        """Log system metrics to system.csv."""
        filepath = self.run_dir / "system.csv"
        timestamp = time.time()

        file_exists = filepath.exists()
        with open(filepath, "a") as f:
            if not file_exists:
                f.write(
                    "timestamp,ram_used_gb,ram_total_gb,"
                    "gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct\n"
                )

            gpu_mem_used = "" if gpu_mem_used_gb is None else f"{gpu_mem_used_gb:.2f}"
            gpu_mem_total = (
                "" if gpu_mem_total_gb is None else f"{gpu_mem_total_gb:.2f}"
            )
            gpu_util = "" if gpu_util_pct is None else f"{gpu_util_pct:.1f}"

            f.write(
                f"{timestamp:.6f},{ram_used_gb:.2f},{ram_total_gb:.2f},"
                f"{gpu_mem_used},{gpu_mem_total},{gpu_util}\n"
            )

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        """Buffer a structured example payload for later writing."""
        record = {
            "step": step,
            "timestamp": time.time(),
            "data": data,
        }
        if name not in self._example_buffer:
            self._example_buffer[name] = []
        self._example_buffer[name].append(record)
        self._maybe_flush()

    def log_confusion_matrix(self, name: str, cm: ConfusionMatrix, step: int) -> None:
        """Buffer a confusion matrix for later writing.

        Parameters
        ----------
        name : str
            Stream name (e.g., "eval/cm").
        cm : ConfusionMatrix
            The matrix and class labels to log.
        step : int
            The training step.
        """
        record = {
            "step": step,
            "timestamp": time.time(),
            "labels": cm.labels,
            "matrix": cm.matrix,
        }
        if name not in self._confusion_buffer:
            self._confusion_buffer[name] = []
        self._confusion_buffer[name].append(record)
        self._maybe_flush()

    def log_chart(self, name: str, chart: Chart, step: int) -> None:
        """Buffer a chart for later writing.

        Parameters
        ----------
        name : str
            Stream name (e.g., "eval/roc").
        chart : Chart
            The points and axis names to log.
        step : int
            The training step.
        """
        record = {
            "step": step,
            "timestamp": time.time(),
            "x_axis": chart.axis_names[0],
            "y_axis": chart.axis_names[1],
            "points": [[x, y] for x, y in chart.points],
        }
        if name not in self._chart_buffer:
            self._chart_buffer[name] = []
        self._chart_buffer[name].append(record)
        self._maybe_flush()

    def close(self) -> None:
        """Flush remaining data and close any open file handles."""
        self.flush()
        for f in self._metric_files.values():
            if hasattr(f, "close"):
                f.close()
        self._metric_files.clear()

    def list_metric_names(self) -> list[str]:
        """List all metric names by scanning the metrics/ directory."""
        metrics_dir = self.run_dir / "metrics"
        if not metrics_dir.exists():
            return []
        names = []
        for csv_file in sorted(metrics_dir.rglob("*.csv")):
            relative = csv_file.relative_to(metrics_dir)
            names.append(relative.with_suffix("").as_posix())
        return names

    def read_metric(self, name: str) -> list[MetricPoint]:
        """Read all data points for a named metric from its CSV file."""
        relative_path = sanitize_metric_name(name) + ".csv"
        filepath = self.run_dir / "metrics" / relative_path
        if not filepath.exists():
            raise FileNotFoundError(f"Metric '{name}' not found at {filepath}")
        return parse_metric_csv(filepath.read_text())

    def read_system_metrics(self) -> list[SystemMetricPoint]:
        """Read system metrics from system.csv."""
        filepath = self.run_dir / "system.csv"
        if not filepath.exists():
            return []
        return parse_system_csv(filepath.read_text())

    def list_example_names(self) -> list[str]:
        """List all example names by scanning the examples/ directory."""
        examples_dir = self.run_dir / "examples"
        if not examples_dir.exists():
            return []
        names = []
        for jsonl_file in sorted(examples_dir.rglob("*.jsonl")):
            relative = jsonl_file.relative_to(examples_dir)
            names.append(relative.with_suffix("").as_posix())
        return names

    def read_examples(self, name: str) -> list[ExampleRecord]:
        """Read all examples for a named example stream from its JSONL file."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "examples" / relative_path
        if not filepath.exists():
            raise FileNotFoundError(f"Examples '{name}' not found at {filepath}")
        return parse_examples_jsonl(filepath.read_text())

    def list_confusion_matrix_names(self) -> list[str]:
        """List all confusion matrix stream names by scanning confusion_matrices/."""
        cm_dir = self.run_dir / "confusion_matrices"
        if not cm_dir.exists():
            return []
        names = []
        for jsonl_file in sorted(cm_dir.rglob("*.jsonl")):
            relative = jsonl_file.relative_to(cm_dir)
            names.append(relative.with_suffix("").as_posix())
        return names

    def read_confusion_matrix(self, name: str) -> list[ConfusionMatrixRecord]:
        """Read all confusion matrix records for a named stream."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "confusion_matrices" / relative_path
        if not filepath.exists():
            raise FileNotFoundError(
                f"Confusion matrix '{name}' not found at {filepath}"
            )
        return parse_confusion_jsonl(filepath.read_text())

    def list_chart_names(self) -> list[str]:
        """List all chart stream names by scanning charts/."""
        chart_dir = self.run_dir / "charts"
        if not chart_dir.exists():
            return []
        names = []
        for jsonl_file in sorted(chart_dir.rglob("*.jsonl")):
            relative = jsonl_file.relative_to(chart_dir)
            names.append(relative.with_suffix("").as_posix())
        return names

    def read_chart(self, name: str) -> list[ChartRecord]:
        """Read all chart records for a named stream."""
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "charts" / relative_path
        if not filepath.exists():
            raise FileNotFoundError(f"Chart '{name}' not found at {filepath}")
        return parse_chart_jsonl(filepath.read_text())

    def read_checkpoints(self) -> list[Checkpoint]:
        """Read the run's checkpoint index from ``checkpoints.json``."""
        filepath = self.run_dir / "checkpoints.json"
        if not filepath.exists():
            return []
        return parse_checkpoints_json(filepath.read_text())

    def record_checkpoint(self, entry: dict[str, Any]) -> None:
        """Merge one checkpoint entry into the local ``checkpoints.json`` index.

        Mirrors the S3-side index update: an existing entry for the same
        step is replaced, otherwise the entry is appended, and the index
        stays sorted by step.

        Parameters
        ----------
        entry : dict[str, Any]
            Index entry with ``step``, ``timestamp``, and ``files`` keys, as
            produced by :meth:`extty.s3.S3Storage.save_checkpoint`.
        """
        filepath = self.run_dir / "checkpoints.json"
        existing: list[dict[str, Any]] = []
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text())
            except json.JSONDecodeError:
                existing = []
        if entry["step"] in {e["step"] for e in existing}:
            existing = [entry if e["step"] == entry["step"] else e for e in existing]
        else:
            existing.append(entry)
        existing.sort(key=lambda e: e["step"])
        filepath.write_text(json.dumps(existing, indent=2))
