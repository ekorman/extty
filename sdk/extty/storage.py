"""File I/O, CSV writing, and path management for extty."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def get_runs_dir() -> Path:
    """Get the default runs directory (~/.ex/runs/)."""
    return Path.home() / ".extty" / "runs"


def get_models_dir() -> Path:
    """Get the default models directory (~/.ex/models/)."""
    return Path.home() / ".extty" / "models"


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
class CheckpointFile:
    name: str
    size_bytes: int


@dataclass(frozen=True)
class Checkpoint:
    step: int
    files: list[CheckpointFile]


def sanitize_metric_name(name: str) -> str:
    """
    Sanitize a metric name for use as a file path.

    Preserves forward slashes as directory separators.
    Sanitizes each path component individually.
    """
    parts = name.split("/")
    sanitized_parts = [re.sub(r"[^\w\-.]", "_", part) for part in parts]
    return "/".join(sanitized_parts)


def generate_random_name() -> str:
    """Generate a unique run name with timestamp and random suffix."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = os.urandom(2).hex()
    return f"{timestamp}_{suffix}"


@dataclass
class ModelMeta:
    """Metadata for a model."""

    project: str
    model_name: str
    model_config: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMeta":
        return cls(
            project=data["project"],
            model_name=data["model_name"],
            model_config=data.get("model_config", {}),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


def log_model_evaluation(
    project: str,
    model: str,
    name: str,
    *,
    metrics: dict[str, float] | None = None,
    examples: list[dict[str, Any]] | None = None,
    model_config: dict[str, Any] | None = None,
    eval_config: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> None:
    """
    Log an evaluation for a model.

    Parameters
    ----------
    project : str
        Name of the project.
    model : str
        Name of the model.
    name : str
        Name of this evaluation (e.g., "gsm8k", "humaneval").
    metrics : dict[str, float], optional
        Evaluation metrics.
    examples : list[dict[str, str]], optional
        Sample outputs.
    model_config : dict[str, Any], optional
        Model configuration (stored on model's meta.json).
    eval_config : dict[str, Any], optional
        Evaluation configuration (stored with evaluation).
    started_at : str, optional
        ISO timestamp when the evaluation started.
    finished_at : str, optional
        ISO timestamp when the evaluation finished.
    """
    now = datetime.now().isoformat()

    model_dir = get_models_dir() / project / model
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = model_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)

    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = ModelMeta.from_dict(json.load(f))
        if model_config is not None:
            existing_meta.model_config = model_config
        existing_meta.updated_at = now
        meta = existing_meta
    else:
        meta = ModelMeta(
            project=project,
            model_name=model,
            model_config=model_config or {},
            created_at=now,
            updated_at=now,
        )

    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    eval_data: dict[str, Any] = {"logged_at": now}
    if started_at is not None:
        eval_data["started_at"] = started_at
    if finished_at is not None:
        eval_data["finished_at"] = finished_at
    if eval_config is not None:
        eval_data["config"] = eval_config
    if metrics is not None:
        eval_data["metrics"] = metrics
    if examples is not None:
        eval_data["examples"] = examples

    eval_path = evaluations_dir / f"{sanitize_metric_name(name)}.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)


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
    """Handles all file I/O for a single run."""

    run_dir: Path
    _readonly: bool = field(default=False, repr=False)
    _metric_files: dict[str, Any] = field(default_factory=dict, repr=False)
    _system_file: Any = field(default=None, repr=False)
    _buffer: list[tuple[str, int, float, float]] = field(
        default_factory=list, repr=False
    )
    _buffer_size: int = 10
    _last_flush: float = field(default_factory=time.time, repr=False)
    _flush_interval: float = 1.0

    def __post_init__(self) -> None:
        if not self._readonly:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "metrics").mkdir(exist_ok=True)
            (self.run_dir / "examples").mkdir(exist_ok=True)

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

        should_flush = (
            len(self._buffer) >= self._buffer_size
            or (time.time() - self._last_flush) >= self._flush_interval
        )
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered metrics to disk."""
        if not self._buffer:
            return

        metrics_by_name: dict[str, list[tuple[int, float, float]]] = {}
        for name, step, timestamp, value in self._buffer:
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append((step, timestamp, value))

        for name, values in metrics_by_name.items():
            self._write_metric_batch(name, values)

        self._buffer.clear()
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
        """Log a structured example payload to a JSONL file."""
        timestamp = time.time()
        relative_path = sanitize_metric_name(name) + ".jsonl"
        filepath = self.run_dir / "examples" / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "step": step,
            "timestamp": timestamp,
            "data": data,
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

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
        points = []
        with open(filepath) as f:
            f.readline()  # skip header
            for line in f:
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

    def read_system_metrics(self) -> list[SystemMetricPoint]:
        """Read system metrics from system.csv."""
        filepath = self.run_dir / "system.csv"
        if not filepath.exists():
            return []
        points = []
        with open(filepath) as f:
            f.readline()  # skip header
            for line in f:
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
        records = []
        with open(filepath) as f:
            for line in f:
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

    def list_checkpoints():
        pass
