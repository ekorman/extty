"""Query API for reading stored run data."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from extty.storage import (
    ExampleRecord,
    MetaData,
    MetricPoint,
    RunStorage,
    SystemMetricPoint,
    get_runs_dir,
)


@dataclass
class RunData:
    """
    Read-only view of a stored run's data.

    Metadata is loaded eagerly. Metric data, system metrics, and examples
    are loaded lazily on each access (not cached).

    Attributes
    ----------
    project : str
        Project name.
    name : str
        Run name.
    config : dict[str, Any]
        Run configuration/hyperparameters.
    started_at : str
        ISO timestamp when the run started.
    finished_at : str or None
        ISO timestamp when the run finished, or None if still running.
    status : str
        Run status ("running" or "completed").
    """

    project: str
    name: str
    config: dict[str, Any]
    started_at: str
    finished_at: str | None
    status: str
    _storage: RunStorage = field(repr=False)

    @property
    def metric_names(self) -> list[str]:
        """List all available metric names for this run."""
        return self._storage.list_metric_names()

    def metric(self, name: str) -> list[MetricPoint]:
        """
        Load all data points for a specific metric.

        Parameters
        ----------
        name : str
            Metric name (e.g., "train/loss").

        Returns
        -------
        list[MetricPoint]
            List of MetricPoint(step, timestamp, value), sorted by step.

        Raises
        ------
        FileNotFoundError
            If the metric does not exist.
        """
        return self._storage.read_metric(name)

    @property
    def system_metrics(self) -> list[SystemMetricPoint]:
        """
        Load system metric samples for this run.

        Returns
        -------
        list[SystemMetricPoint]
            System metric samples sorted by timestamp,
            or empty list if no system metrics were recorded.
        """
        return self._storage.read_system_metrics()

    @property
    def example_names(self) -> list[str]:
        """List all available example names for this run."""
        return self._storage.list_example_names()

    def examples(self, name: str) -> list[ExampleRecord]:
        """
        Load all examples for a specific name.

        Parameters
        ----------
        name : str
            Example name (e.g., "val/example").

        Returns
        -------
        list[ExampleRecord]
            List of ExampleRecord(step, timestamp, data), sorted by step.

        Raises
        ------
        FileNotFoundError
            If the example file does not exist.
        """
        return self._storage.read_examples(name)

    @property
    def duration_seconds(self) -> float | None:
        """Duration of the run in seconds, or None if not finished."""
        if self.finished_at is None:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.finished_at)
        return (end - start).total_seconds()


def _run_data_from_meta(meta: MetaData, storage: RunStorage) -> RunData:
    return RunData(
        project=meta.project,
        name=meta.run_name,
        config=meta.config,
        started_at=meta.started_at,
        finished_at=meta.finished_at,
        status=meta.status,
        _storage=storage,
    )


def get_runs(project: str | None = None) -> list[RunData]:
    """
    List stored runs with full metadata loaded.

    Parameters
    ----------
    project : str, optional
        Filter to runs in this project only. If None, returns all runs.

    Returns
    -------
    list[RunData]
        List of RunData objects with metadata loaded, sorted by started_at
        (most recent first).
    """
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    result = []
    for proj_dir in runs_dir.iterdir():
        if not proj_dir.is_dir():
            continue
        if project is not None and proj_dir.name != project:
            continue
        for run_dir in proj_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not (run_dir / "meta.json").exists():
                continue
            try:
                storage = RunStorage.open_readonly(run_dir)
                meta = storage.read_meta()
                if meta is None:
                    continue
                result.append(_run_data_from_meta(meta, storage))
            except (json.JSONDecodeError, KeyError):
                warnings.warn(f"Skipping run with invalid meta.json: {run_dir}")
                continue

    result.sort(key=lambda r: r.started_at, reverse=True)
    return result


def get_run(project: str, name: str) -> RunData:
    """
    Get a single stored run by project and name.

    Parameters
    ----------
    project : str
        Project name.
    name : str
        Run name.

    Returns
    -------
    RunData
        The run data object with metadata loaded.

    Raises
    ------
    FileNotFoundError
        If the run does not exist.
    """
    project_dir = project if project else "_default"
    run_dir = get_runs_dir() / project_dir / name
    if not run_dir.exists() or not (run_dir / "meta.json").exists():
        raise FileNotFoundError(f"Run '{project}/{name}' not found at {run_dir}")

    storage = RunStorage.open_readonly(run_dir)
    meta = storage.read_meta()
    if meta is None:
        raise FileNotFoundError(f"Run '{project}/{name}' has no valid meta.json")

    return _run_data_from_meta(meta, storage)
