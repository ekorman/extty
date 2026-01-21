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
    """Get the default runs directory (~/.extty/runs/)."""
    return Path.home() / ".ex" / "runs"


def sanitize_metric_name(name: str) -> str:
    """
    Sanitize a metric name for use as a file path.

    Preserves forward slashes as directory separators.
    Sanitizes each path component individually.
    """
    parts = name.split("/")
    sanitized_parts = [re.sub(r"[^\w\-.]", "_", part) for part in parts]
    return "/".join(sanitized_parts)


def generate_run_name() -> str:
    """Generate a unique run name with timestamp and random suffix."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = os.urandom(2).hex()
    return f"{timestamp}_{suffix}"


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
    _metric_files: dict[str, Any] = field(default_factory=dict, repr=False)
    _system_file: Any = field(default=None, repr=False)
    _buffer: list[tuple[str, int, float, float]] = field(
        default_factory=list, repr=False
    )
    _buffer_size: int = 10
    _last_flush: float = field(default_factory=time.time, repr=False)
    _flush_interval: float = 1.0

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "examples").mkdir(exist_ok=True)

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

    def log_evaluation(
        self,
        name: str,
        metrics: dict[str, float] | None = None,
        examples: list[dict[str, str]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an evaluation snapshot.

        Parameters
        ----------
        name : str
            Name of this evaluation (e.g., "gsm8k", "humaneval").
        metrics : dict[str, float], optional
            Evaluation metrics.
        examples : list[dict[str, str]], optional
            Sample outputs.
        config : dict[str, Any], optional
            Evaluation configuration.
        """
        evaluations_dir = self.run_dir / "evaluations"
        evaluations_dir.mkdir(exist_ok=True)

        data: dict[str, Any] = {}
        if config is not None:
            data["config"] = config
        if metrics is not None:
            data["metrics"] = metrics
        if examples is not None:
            data["examples"] = examples

        filepath = evaluations_dir / f"{sanitize_metric_name(name)}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        """Flush remaining data and close any open file handles."""
        self.flush()
        for f in self._metric_files.values():
            if hasattr(f, "close"):
                f.close()
        self._metric_files.clear()
