"""Background-thread wrapper for ``StorageSink`` implementations."""

from __future__ import annotations

import queue
import threading
from typing import Any

from extty._logger import log as logger
from extty._sink import StorageSink
from extty.chart import Chart
from extty.confusion import ConfusionMatrix

_HIGH_WATER_MARK = 100_000
_STOP = object()


class AsyncSink:
    """
    Run every ``log_*`` call on a background daemon thread.

    Wraps any object implementing the :class:`StorageSink` protocol so that
    ``log_metric``, ``log_example`` and ``log_system`` return as soon as the
    work has been enqueued. The wrapped sink's existing batching and flush
    cadence run on the worker thread, where blocking is harmless.

    Parameters
    ----------
    inner : StorageSink
        The sink to dispatch to.
    name : str
        Thread name, useful for debugging.

    Notes
    -----
    The queue is unbounded so that the producer (training thread) never blocks.
    A one-shot warning is emitted if the backlog grows past ``100_000`` items,
    which would indicate the writer is unable to keep up with production.

    The worker thread is a daemon, so callers MUST invoke :meth:`close` (or
    :meth:`flush` followed by :meth:`close`) to guarantee durability. At
    interpreter exit, daemon threads are killed without draining; an
    ``atexit`` handler registered by :func:`extty.init` finishes the active
    run as a safety net.
    """

    def __init__(self, inner: StorageSink, *, name: str = "extty-writer") -> None:
        self._inner = inner
        self._queue: queue.Queue[Any] = queue.Queue()
        self._high_water_warned = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def log_metric(self, name: str, value: float, step: int) -> None:
        self._enqueue(("metric", name, value, step))

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        self._enqueue(("example", name, data, step))

    def log_confusion_matrix(self, name: str, cm: ConfusionMatrix, step: int) -> None:
        self._enqueue(("confusion", name, cm, step))

    def log_chart(self, name: str, chart: Chart, step: int) -> None:
        self._enqueue(("chart", name, chart, step))

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        self._enqueue(
            (
                "system",
                ram_used_gb,
                ram_total_gb,
                gpu_mem_used_gb,
                gpu_mem_total_gb,
                gpu_util_pct,
            )
        )

    def flush(self) -> None:
        """Block until every item enqueued so far has been replayed."""
        self._queue.join()
        self._inner.flush()

    def close(self) -> None:
        """Drain the queue, join the worker thread, then close the inner sink."""
        if not self._thread.is_alive():
            self._inner.close()
            return
        self._queue.put(_STOP)
        self._thread.join()
        self._inner.close()

    def _enqueue(self, item: tuple[Any, ...]) -> None:
        self._queue.put(item)
        if not self._high_water_warned and self._queue.qsize() > _HIGH_WATER_MARK:
            self._high_water_warned = True
            logger.warning(
                "extty AsyncSink(%s) is falling behind: %d items pending",
                type(self._inner).__name__,
                self._queue.qsize(),
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    try:
                        self._inner.flush()
                    except Exception:
                        logger.exception(
                            "extty AsyncSink worker failed final flush of %s",
                            type(self._inner).__name__,
                        )
                    return
                try:
                    self._dispatch(item)
                except Exception:
                    logger.exception(
                        "extty AsyncSink worker swallowed exception dispatching to %s",
                        type(self._inner).__name__,
                    )
            finally:
                self._queue.task_done()

    def _dispatch(self, item: tuple[Any, ...]) -> None:
        tag = item[0]
        if tag == "metric":
            _, name, value, step = item
            self._inner.log_metric(name, value, step)
        elif tag == "example":
            _, name, data, step = item
            self._inner.log_example(name, data, step)
        elif tag == "confusion":
            _, name, cm, step = item
            self._inner.log_confusion_matrix(name, cm, step)
        elif tag == "chart":
            _, name, chart, step = item
            self._inner.log_chart(name, chart, step)
        elif tag == "system":
            _, ram_u, ram_t, gpu_u, gpu_t, gpu_p = item
            self._inner.log_system(ram_u, ram_t, gpu_u, gpu_t, gpu_p)
        else:
            raise AssertionError(f"unknown AsyncSink dispatch tag: {tag!r}")
