"""Background thread for system metrics collection."""

from __future__ import annotations

import threading
from typing import Protocol

import psutil

class SystemMetricSink(Protocol):
    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None: ...

try:
    import pynvml as _pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    _pynvml = None  # type: ignore[assignment]
    PYNVML_AVAILABLE = False


class SystemMonitor:
    """
    Background thread that samples system metrics periodically.

    Collects RAM usage and optionally GPU metrics if pynvml is available.
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

        if self._gpu_initialized and _pynvml is not None:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass
            self._gpu_initialized = False

    def _init_gpu(self) -> None:
        """Initialize pynvml if available."""
        if not PYNVML_AVAILABLE or _pynvml is None:
            return

        try:
            _pynvml.nvmlInit()
            device_count = _pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_initialized = True
        except Exception:
            self._gpu_initialized = False

    def _run(self) -> None:
        """Main loop for the monitoring thread."""
        while not self._stop_event.wait(self._interval):
            self._sample()
        self._sample()

    def _sample(self) -> None:
        """Take a single sample of system metrics."""
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)

        gpu_mem_used_gb: float | None = None
        gpu_mem_total_gb: float | None = None
        gpu_util_pct: float | None = None

        if (
            self._gpu_initialized
            and self._gpu_handle is not None
            and _pynvml is not None
        ):
            try:
                mem_info = _pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used_gb = mem_info.used / (1024**3)
                gpu_mem_total_gb = mem_info.total / (1024**3)

                util = _pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util_pct = float(util.gpu)
            except Exception:
                pass

        self._storage.log_system(
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_mem_used_gb=gpu_mem_used_gb,
            gpu_mem_total_gb=gpu_mem_total_gb,
            gpu_util_pct=gpu_util_pct,
        )
