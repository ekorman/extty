"""Query API for reading stored run data."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, overload

from extty.storage import (
    ChartRecord,
    Checkpoint,
    ConfusionMatrixRecord,
    ExampleRecord,
    ImageRecord,
    MetaData,
    MetricPoint,
    RunStorage,
    RunStorageReader,
    SystemMetricPoint,
    get_run_dir,
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
    _storage: RunStorageReader = field(repr=False)

    @property
    def metric_names(self) -> list[str]:
        """List all available metric names for this run."""
        return self._storage.list_metric_names()

    @overload
    def metric(self, name: str) -> list[MetricPoint]: ...

    @overload
    def metric(self, name: str, step: int) -> MetricPoint | None: ...

    def metric(
        self, name: str, step: int | None = None
    ) -> list[MetricPoint] | MetricPoint | None:
        """
        Load data points for a specific metric.

        Parameters
        ----------
        name : str
            Metric name (e.g., "train/loss").
        step : int, optional
            If given, return only the data point logged at this step rather
            than the full history.

        Returns
        -------
        list[MetricPoint] or MetricPoint or None
            When ``step`` is omitted, the list of MetricPoint(step, timestamp,
            value) sorted by step. When ``step`` is given, the matching
            MetricPoint, or None if no point was logged at that step.

        Raises
        ------
        FileNotFoundError
            If the metric does not exist.
        """
        points = self._storage.read_metric(name)
        if step is None:
            return points
        return next((p for p in points if p.step == step), None)

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
    def confusion_matrix_names(self) -> list[str]:
        """List all available confusion matrix names for this run."""
        return self._storage.list_confusion_matrix_names()

    def confusion_matrix(self, name: str) -> list[ConfusionMatrixRecord]:
        """
        Load all confusion matrix records for a specific name.

        Parameters
        ----------
        name : str
            Confusion matrix name (e.g., "eval/cm").

        Returns
        -------
        list[ConfusionMatrixRecord]
            List of ConfusionMatrixRecord(step, timestamp, labels, matrix),
            sorted by step in the order written.

        Raises
        ------
        FileNotFoundError
            If the confusion matrix file does not exist.
        """
        return self._storage.read_confusion_matrix(name)

    @property
    def chart_names(self) -> list[str]:
        """List all available chart names for this run."""
        return self._storage.list_chart_names()

    def chart(self, name: str) -> list[ChartRecord]:
        """
        Load all chart records for a specific name.

        Parameters
        ----------
        name : str
            Chart name (e.g., "eval/roc").

        Returns
        -------
        list[ChartRecord]
            List of ChartRecord(step, timestamp, x_axis, y_axis, points),
            sorted by step in the order written.

        Raises
        ------
        FileNotFoundError
            If the chart file does not exist.
        """
        return self._storage.read_chart(name)

    @property
    def image_names(self) -> list[str]:
        """List all available image stream names for this run."""
        return self._storage.list_image_names()

    def images(self, name: str) -> list[ImageRecord]:
        """
        Load all image records for a specific name.

        Parameters
        ----------
        name : str
            Image stream name (e.g., "val/detections").

        Returns
        -------
        list[ImageRecord]
            List of ImageRecord(step, timestamp, file, width, height, caption),
            sorted by step with one record per step (latest write wins).

        Raises
        ------
        FileNotFoundError
            If the image stream does not exist.
        """
        return self._storage.read_images(name)

    def image_bytes(self, record: ImageRecord | str) -> bytes:
        """
        Load the PNG bytes for a logged image.

        Parameters
        ----------
        record : ImageRecord or str
            An :class:`ImageRecord` from :meth:`images`, or the record's
            ``file`` path relative to the run's ``images/`` directory.

        Returns
        -------
        bytes
            The PNG-encoded image.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """
        file = record.file if isinstance(record, ImageRecord) else record
        return self._storage.read_image_bytes(file)

    @property
    def checkpoints(self) -> list[Checkpoint]:
        """
        List the checkpoints saved for this run.

        Reads the run's ``checkpoints.json`` index.

        Returns
        -------
        list[Checkpoint]
            Checkpoints sorted by step, or an empty list if none were saved.
        """
        return self._storage.read_checkpoints()

    @property
    def duration_seconds(self) -> float | None:
        """Duration of the run in seconds, or None if not finished."""
        if self.finished_at is None:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.finished_at)
        return (end - start).total_seconds()


def _run_data_from_meta(meta: MetaData, storage: RunStorageReader) -> RunData:
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


def get_run(project: str, name: str, *, local_only: bool = False) -> RunData:
    """
    Get a single stored run by project and name.

    Looks on the local filesystem first. If the run is not present locally
    and *local_only* is False, falls back to reading the run directly from
    S3 using the globally-configured S3 credentials (env vars or
    ``~/.extty/s3/config.toml``); no data is downloaded to disk in that
    case — reads stream straight from S3.

    Parameters
    ----------
    project : str
        Project name.
    name : str
        Run name.
    local_only : bool, default False
        If True, only consult the local runs directory and never reach
        out to S3.

    Returns
    -------
    RunData
        The run data object with metadata loaded.

    Raises
    ------
    FileNotFoundError
        If the run is not found locally, and either *local_only* is True,
        no S3 configuration is available, or the run is also missing in S3.
    """
    run_dir = get_run_dir(project, name)
    if run_dir.exists() and (run_dir / "meta.json").exists():
        storage = RunStorage.open_readonly(run_dir)
        meta = storage.read_meta()
        if meta is None:
            raise FileNotFoundError(f"Run '{project}/{name}' has no valid meta.json")
        return _run_data_from_meta(meta, storage)

    if local_only:
        raise FileNotFoundError(f"Run '{project}/{name}' not found at {run_dir}")

    from extty.s3 import S3Config, S3RunReader

    s3_config = S3Config.load()
    if s3_config is None:
        raise FileNotFoundError(
            f"Run '{project}/{name}' not found at {run_dir} and no S3 configuration "
            "is available (set EXTTY_S3_BUCKET or ~/.extty/s3/config.toml)."
        )

    reader = S3RunReader(config=s3_config, project=project or "_default", run_name=name)
    meta = reader.read_meta()
    if meta is None:
        raise FileNotFoundError(
            f"Run '{project}/{name}' not found locally or in S3 at "
            f"s3://{s3_config.bucket}/{reader._s3_key()}"
        )
    return _run_data_from_meta(meta, reader)
