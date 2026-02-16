"""Background thread for system metrics collection."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import psutil
import pynvml as _pynvml

logger = logging.getLogger(__name__)


class SystemMetricSink(Protocol):
    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None: ...


class SystemMonitor:
    """
    Background thread that samples system metrics periodically.

    Collects RAM usage and GPU metrics (when a GPU is present).
    """

    def __init__(
        self,
        storage: SystemMetricSink,
        interval: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        storage : RunStorage
            Storage instance to write metrics to.
        interval : float
            Seconds between samples.
        """
        self._storage = storage
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_initialized = False
        self._gpu_handle = None
        self._gpu_sample_error_logged = False

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None:
            return

        self._init_gpu()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._gpu_initialized:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass
            self._gpu_initialized = False

    def _init_gpu(self) -> None:
        """Initialize pynvml if a GPU is present."""
        try:
            _pynvml.nvmlInit()
            device_count = _pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_initialized = True
            else:
                logger.warning("No NVIDIA GPUs detected — GPU metrics disabled")
        except Exception:
            logger.warning(
                "Failed to initialize pynvml — GPU metrics disabled", exc_info=True
            )
            self._gpu_initialized = False

    def _run(self) -> None:
        """Main loop for the monitoring thread."""
        self._sample()
        while not self._stop_event.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        """Take a single sample of system metrics."""
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)

        gpu_mem_used_gb: float | None = None
        gpu_mem_total_gb: float | None = None
        gpu_util_pct: float | None = None

        if self._gpu_initialized and self._gpu_handle is not None:
            try:
                mem_info = _pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used_gb = mem_info.used / (1024**3)
                gpu_mem_total_gb = mem_info.total / (1024**3)

                util = _pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util_pct = float(util.gpu)
            except Exception:
                if not self._gpu_sample_error_logged:
                    logger.warning("Error reading GPU metrics", exc_info=True)
                    self._gpu_sample_error_logged = True

        self._storage.log_system(
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_mem_used_gb=gpu_mem_used_gb,
            gpu_mem_total_gb=gpu_mem_total_gb,
            gpu_util_pct=gpu_util_pct,
        )
