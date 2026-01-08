"""Run class - manages single run state."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from extty.storage import (
    MetaData,
    RunStorage,
    generate_run_name,
    get_runs_dir,
)
from extty.system_monitor import SystemMonitor


class Run:
    """
    Represents a single experiment run.

    Manages logging metrics, system monitoring, and storage.
    """

    def __init__(
        self,
        project: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        system_metrics: bool = True,
    ) -> None:
        self.project = project
        self.name = name or generate_run_name()
        self.config = config or {}
        self._system_metrics_enabled = system_metrics

        run_dir = get_runs_dir() / self.name
        self._storage = RunStorage(run_dir=run_dir)

        self._meta = MetaData(
            project=project,
            run_name=self.name,
            config=self.config,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._storage.write_meta(self._meta)

        self._system_monitor: SystemMonitor | None = None
        if system_metrics:
            self._system_monitor = SystemMonitor(self._storage)
            self._system_monitor.start()

        self._lock = threading.Lock()
        self._finished = False

    def log(
        self,
        metrics: dict[str, float] | str,
        value: dict[str, Any] | None = None,
        *,
        step: int,
    ) -> None:
        """
        Log metrics or structured examples for the current step.

        Parameters
        ----------
        metrics : dict[str, float] | str
            Dictionary of metric names to values, or the example name.
        value : dict[str, Any] | None
            Example payload when logging structured examples.
        step : int
            The current training step.

        Raises
        ------
        RuntimeError
            If the run has already been finished.
        """
        with self._lock:
            if self._finished:
                raise RuntimeError("Cannot log to a finished run.")
            if isinstance(metrics, str):
                if value is None or not isinstance(value, dict):
                    raise TypeError("Example logging requires a dict payload.")
                self._storage.log_example(metrics, value, step)
                return
            if value is not None:
                raise TypeError("Metric logging does not accept an example payload.")
            for name, metric_value in metrics.items():
                self._storage.log_metric(name, metric_value, step)

    def finish(self) -> None:
        """
        Finish the run.

        Stops system monitoring, flushes all data, and marks run complete.
        """
        with self._lock:
            if self._finished:
                return
            self._finished = True

        if self._system_monitor is not None:
            self._system_monitor.stop()
            self._system_monitor = None

        self._storage.flush()

        self._meta.finished_at = datetime.now(timezone.utc).isoformat()
        self._meta.status = "completed"
        self._storage.write_meta(self._meta)
        self._storage.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.finish()

    @property
    def run_dir(self) -> str:
        """Return the path to this run's directory."""
        return str(self._storage.run_dir)
