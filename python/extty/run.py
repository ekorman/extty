"""Run class - manages single run state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from extty.storage import (
    MetaData,
    RunStorage,
    generate_run_name,
    get_runs_dir,
)
from extty.system_monitor import SystemMonitor
from extty.server import QueueStorage, ServerManager, ServerSettings, ServerInfo


@dataclass(frozen=True)
class ServerConfig:
    enabled: bool
    settings: ServerSettings | None = None


class StorageSink(Protocol):
    def log_metric(self, name: str, value: float, step: int) -> None: ...

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None: ...

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class FinishableStorage(Protocol):
    def finish(self, finished_at: str, status: str) -> None: ...


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
        server: ServerConfig | None = None,
    ) -> None:
        self.project = project
        self.name = name or generate_run_name()
        self.config = config or {}
        self._system_metrics_enabled = system_metrics

        self._server_manager: ServerManager | None = None
        self._server_info: ServerInfo | None = None
        self._storage: StorageSink
        self._meta: MetaData | None = None

        if server is not None and server.enabled:
            settings = server.settings or ServerSettings()
            self._server_manager = ServerManager(settings)
            self._server_manager.start()
            self._server_info = self._server_manager.info
            started_at = datetime.now(timezone.utc).isoformat()
            self._storage = QueueStorage(
                self._server_manager,
                run_name=self.name,
                project=project,
                config=self.config,
                started_at=started_at,
            )
            self._meta = MetaData(
                project=project,
                run_name=self.name,
                config=self.config,
                started_at=started_at,
            )
        else:
            project_dir = project if project else "_default"
            run_dir = get_runs_dir() / project_dir / self.name
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
        metrics: dict[str, Any],
        *,
        step: int,
    ) -> None:
        """
        Log metrics or structured examples for the current step.

        Parameters
        ----------
        metrics : dict[str, Any]
            Dictionary of metric names to values. Values can be:
            - float/int: logged as metric
            - Example/BatchExample: logged as structured example
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
            for name, value in metrics.items():
                if hasattr(value, "to_dict"):
                    self._storage.log_example(name, value.to_dict(), step)
                else:
                    self._storage.log_metric(name, float(value), step)

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

        if self._meta is not None:
            self._meta.finished_at = datetime.now(timezone.utc).isoformat()
            self._meta.status = "completed"
            if isinstance(self._storage, RunStorage):
                self._storage.write_meta(self._meta)
            if isinstance(self._storage, FinishableStorage):
                self._storage.finish(self._meta.finished_at, self._meta.status)
        self._storage.close()
        if self._server_manager is not None:
            self._server_manager.stop()
            self._server_manager = None

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.finish()

    @property
    def run_dir(self) -> str:
        """Return the path to this run's directory."""
        if isinstance(self._storage, RunStorage):
            return str(self._storage.run_dir)
        return ""

    @property
    def server_info(self) -> ServerInfo | None:
        return self._server_info
