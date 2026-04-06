"""Run class - manages single run state."""

from __future__ import annotations
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from extty.storage import (
    MetaData,
    RunStorage,
    generate_random_name,
    get_runs_dir,
)
from extty.system_monitor import SystemMonitor
from extty.s3 import S3Config, S3Storage
from extty.artifact import ArtifactMeta, save_artifact as _save_artifact


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


class MultiSink:
    """Dispatches storage operations to multiple sinks."""

    def __init__(
        self, primary: StorageSink, secondary: S3Storage | None = None
    ) -> None:
        self._primary = primary
        self._secondary = secondary

    def log_metric(self, name: str, value: float, step: int) -> None:
        self._primary.log_metric(name, value, step)
        if self._secondary:
            self._secondary.log_metric(name, value, step)

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        self._primary.log_example(name, data, step)
        if self._secondary:
            self._secondary.log_example(name, data, step)

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        self._primary.log_system(
            ram_used_gb, ram_total_gb, gpu_mem_used_gb, gpu_mem_total_gb, gpu_util_pct
        )
        if self._secondary:
            self._secondary.log_system(
                ram_used_gb,
                ram_total_gb,
                gpu_mem_used_gb,
                gpu_mem_total_gb,
                gpu_util_pct,
            )

    def flush(self) -> None:
        self._primary.flush()
        if self._secondary:
            self._secondary.flush()

    def close(self) -> None:
        self._primary.close()
        if self._secondary:
            self._secondary.close()


@runtime_checkable
class FinishableStorage(Protocol):
    def finish(self, finished_at: str, status: str) -> None: ...


def _detect_gpu() -> tuple[str, int | None] | None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            vram_mb = total // (1024 * 1024)
            return name, vram_mb
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return f"Apple MPS ({result.stdout.strip()})", None
            return "Apple MPS", None
    except ImportError:
        pass
    return None


def _collect_environment() -> dict[str, Any]:
    from extty import __version__

    env: dict[str, Any] = {"_extty_version": __version__}
    git_hash = os.environ.get("EXTTY_GIT_HASH")
    if git_hash:
        env["_git_hash"] = git_hash
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                env["_git_hash"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    run_command = os.environ.get("EXTTY_RUN_COMMAND")
    if run_command:
        env["_run_command"] = run_command

    instance_id = os.environ.get("EXTTY_INSTANCE_ID")
    instance_provider = os.environ.get("EXTTY_INSTANCE_PROVIDER")
    if instance_id and instance_provider:
        env["_instance_id"] = f"{instance_provider}:{instance_id}"

    gpu_info = _detect_gpu()
    if gpu_info:
        env["_gpu"] = gpu_info[0]
        if gpu_info[1] is not None:
            env["_gpu_vram_mb"] = gpu_info[1]

    return env


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
        s3_config: S3Config | None = None,
    ) -> None:
        self.project = project
        self.name = name or generate_random_name()
        self.config = {**_collect_environment(), **(config or {})}
        self._system_metrics_enabled = system_metrics

        self._storage: StorageSink
        self._s3_storage: S3Storage | None = None
        self._meta: MetaData | None = None

        if s3_config is None:
            s3_config = S3Config.load()
        if s3_config is not None:
            self._s3_storage = S3Storage(s3_config, project, self.name)

        project_dir = project if project else "_default"
        run_dir = get_runs_dir() / project_dir / self.name
        local_storage = RunStorage(run_dir=run_dir)
        primary_storage: StorageSink = local_storage
        self._meta = MetaData(
            project=project,
            run_name=self.name,
            config=self.config,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        local_storage.write_meta(self._meta)

        if self._s3_storage and self._meta:
            self._s3_storage.write_meta(self._meta.to_dict())

        self._storage = MultiSink(primary_storage, self._s3_storage)

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

    def save_checkpoint(
        self,
        step: int,
        *,
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
        RuntimeError
            If no S3 storage is configured or the run is finished.
        """
        with self._lock:
            if self._finished:
                raise RuntimeError("Cannot save checkpoint on a finished run.")
        if self._s3_storage is None:
            raise RuntimeError(
                "S3 storage is not configured. "
                "Set EXTTY_S3_BUCKET or provide s3_config to save checkpoints."
            )
        self._s3_storage.save_checkpoint(
            step,
            path=path,
            state_dict=state_dict,
            optimizer_state_dict=optimizer_state_dict,
        )

    def load_checkpoint(
        self,
        step: int,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """
        Load a checkpoint, downloading from S3 if not cached locally.

        Parameters
        ----------
        step : int
            The training step to load.
        load_optimizer : bool, default True
            Whether to include the optimizer state in the result.

        Returns
        -------
        dict[str, Any]
            Contains ``"model_state_dict"`` and optionally
            ``"optimizer_state_dict"``.

        Raises
        ------
        RuntimeError
            If no S3 storage is configured.
        FileNotFoundError
            If the checkpoint step does not exist.
        """
        if self._s3_storage is None:
            raise RuntimeError(
                "S3 storage is not configured. "
                "Set EXTTY_S3_BUCKET or provide s3_config to load checkpoints."
            )
        return self._s3_storage.load_checkpoint(step, load_optimizer=load_optimizer)

    def save_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        s3_config: S3Config | None = None,
    ) -> ArtifactMeta:
        """
        Save an artifact associated with this run.

        Parameters
        ----------
        name : str
            Unique name for this artifact.
        path : str or Path
            Local file or directory to upload.
        description : str
            Human-readable description.
        metadata : dict[str, Any], optional
            User-defined metadata (arbitrary JSON-serializable dict).
        s3_config : S3Config, optional
            S3 configuration. Loaded from environment if not provided.

        Returns
        -------
        ArtifactMeta
            Metadata for the saved artifact.
        """
        return _save_artifact(
            name,
            path,
            description=description,
            metadata=metadata,
            s3_config=s3_config,
            run_project=self.project,
            run_name=self.name,
        )

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
            primary = (
                self._storage._primary
                if isinstance(self._storage, MultiSink)
                else self._storage
            )
            if isinstance(primary, RunStorage):
                primary.write_meta(self._meta)
            if isinstance(primary, FinishableStorage):
                primary.finish(self._meta.finished_at, self._meta.status)
            if self._s3_storage:
                self._s3_storage.write_meta(self._meta.to_dict())

        self._storage.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.finish()

    @property
    def run_dir(self) -> str:
        """Return the path to this run's directory."""
        primary = (
            self._storage._primary
            if isinstance(self._storage, MultiSink)
            else self._storage
        )
        if isinstance(primary, RunStorage):
            return str(primary.run_dir)
        return ""
