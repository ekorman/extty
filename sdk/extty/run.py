"""Run class - manages single run state."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from extty._logger import log as logger
from extty._sink import StorageSink
from extty.artifact import ArtifactMeta
from extty.artifact import save_artifact as _save_artifact
from extty.async_sink import AsyncSink
from extty.chart import Chart
from extty.checkpoints import (
    checkpoint_dir,
    read_local_checkpoint,
    write_checkpoint,
)
from extty.confusion import ConfusionMatrix
from extty.image import Image
from extty.s3 import S3Config, S3Storage
from extty.storage import (
    MetaData,
    RunStorage,
    generate_random_name,
    get_runs_dir,
)
from extty.system_monitor import SystemMonitor

__all__ = ["Run", "NoOpRun", "StorageSink", "MultiSink"]


# ``StorageSink`` moved to :mod:`extty._sink` so the async wrapper can import it
# without a cycle; it stays re-exported here for any caller importing
# ``extty.run.StorageSink``.


class MultiSink:
    """Dispatches storage operations to multiple sinks."""

    def __init__(
        self, primary: StorageSink, secondary: StorageSink | None = None
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

    def log_confusion_matrix(self, name: str, cm: ConfusionMatrix, step: int) -> None:
        self._primary.log_confusion_matrix(name, cm, step)
        if self._secondary:
            self._secondary.log_confusion_matrix(name, cm, step)

    def log_chart(self, name: str, chart: Chart, step: int) -> None:
        self._primary.log_chart(name, chart, step)
        if self._secondary:
            self._secondary.log_chart(name, chart, step)

    def log_image(self, name: str, image: Image, step: int) -> None:
        self._primary.log_image(name, image, step)
        if self._secondary:
            self._secondary.log_image(name, image, step)

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
        self._local_storage = local_storage
        self._meta = MetaData(
            project=project,
            run_name=self.name,
            config=self.config,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        local_storage.write_meta(self._meta)

        if self._s3_storage and self._meta:
            self._s3_storage.write_meta(self._meta.to_dict())

        primary_storage: StorageSink = AsyncSink(
            local_storage, name="extty-writer-local"
        )
        secondary: StorageSink | None = (
            AsyncSink(self._s3_storage, name="extty-writer-s3")
            if self._s3_storage is not None
            else None
        )
        self._storage = MultiSink(primary_storage, secondary)

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
            - ConfusionMatrix: logged as a confusion matrix
            - Chart: logged as a 2D chart of (x, y) points
            - Image: logged as a PNG image
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
            if isinstance(value, ConfusionMatrix):
                self._storage.log_confusion_matrix(name, value, step)
            elif isinstance(value, Chart):
                self._storage.log_chart(name, value, step)
            elif isinstance(value, Image):
                self._storage.log_image(name, value, step)
            elif hasattr(value, "to_dict"):
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
        keep_local: bool = False,
    ) -> None:
        """
        Save a checkpoint to the run directory, and to S3 when configured.

        The checkpoint is always written to ``<run dir>/checkpoints/<step>/``
        first. With S3 configured it is then uploaded, and the local copy is
        removed once the upload succeeds unless ``keep_local`` is set. If the
        upload fails, the local copy is kept.

        Parameters
        ----------
        step : int
            The training step for this checkpoint.
        path : str or None
            Path to an existing checkpoint file to save.
        state_dict : Any or None
            Model state dict to serialize with torch.save.
        optimizer_state_dict : Any or None
            Optimizer state dict to include when using state_dict.
        keep_local : bool, default False
            Keep the local copy after a successful S3 upload.

        Raises
        ------
        RuntimeError
            If the run is finished.
        """
        with self._lock:
            if self._finished:
                raise RuntimeError("Cannot save checkpoint on a finished run.")
        local_dir = checkpoint_dir(self._local_storage.run_dir, step)
        entry = write_checkpoint(
            local_dir,
            step,
            path=path,
            state_dict=state_dict,
            optimizer_state_dict=optimizer_state_dict,
        )
        self._local_storage.record_checkpoint(entry)
        if self._s3_storage is None:
            logger.info("checkpoint step %d: saved to %s", step, local_dir)
            return
        if not self._s3_storage.upload_checkpoint(local_dir, entry):
            logger.error(
                "checkpoint step %d: S3 upload failed, local copy kept at %s",
                step,
                local_dir,
            )
            return
        if not keep_local:
            shutil.rmtree(local_dir)

    def load_checkpoint(
        self,
        step: int,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """
        Load a checkpoint from the run directory, falling back to S3.

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
        FileNotFoundError
            If the checkpoint is neither in the run directory nor in S3.
        """
        local_dir = checkpoint_dir(self._local_storage.run_dir, step)
        local = read_local_checkpoint(local_dir, load_optimizer=load_optimizer)
        if local is not None:
            return local
        if self._s3_storage is None:
            raise FileNotFoundError(
                f"Checkpoint step {step} not found in {local_dir} "
                "and S3 is not configured."
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
        Safe to call twice — the second call short-circuits, which is what
        keeps the :mod:`extty` atexit handler from double-finishing a run
        that the user has already closed.
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
            self._local_storage.write_meta(self._meta)
            if isinstance(self._local_storage, FinishableStorage):
                self._local_storage.finish(self._meta.finished_at, self._meta.status)
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
        return str(self._local_storage.run_dir)


class NoOpRun(Run):
    """A run that silently discards all operations.

    Used on non-main processes in distributed training to prevent
    phantom runs while keeping the API functional.
    """

    def __init__(
        self,
        project: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.name = name or generate_random_name()
        self.config = config or {}
        self._finished = False
        self._lock = threading.Lock()
        self._system_monitor = None
        self._s3_storage = None

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        pass

    def save_checkpoint(
        self,
        step: int,
        *,
        path: str | None = None,
        state_dict: Any = None,
        optimizer_state_dict: Any = None,
        keep_local: bool = False,
    ) -> None:
        pass

    def load_checkpoint(
        self,
        step: int,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "Cannot load checkpoints from a no-op run. "
            "Load checkpoints on the main process (rank 0) only."
        )

    def save_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        s3_config: S3Config | None = None,
    ) -> ArtifactMeta:
        return ArtifactMeta(
            name=name,
            description=description,
            content_type="file",
            created_at="",
            updated_at="",
            total_size_bytes=0,
            files=[],
            metadata=metadata or {},
            run_project=self.project,
            run_name=self.name,
        )

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    @property
    def run_dir(self) -> str:
        return ""

    def __enter__(self) -> NoOpRun:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.finish()
