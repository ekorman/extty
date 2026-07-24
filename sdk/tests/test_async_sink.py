"""Tests for extty.async_sink.AsyncSink."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import extty
from extty.async_sink import _HIGH_WATER_MARK, AsyncSink


@dataclass
class _FakeSink:
    """Records every call made to it. Implements StorageSink by duck typing."""

    metrics: list[tuple[str, float, int]] = field(default_factory=list)
    examples: list[tuple[str, dict[str, Any], int]] = field(default_factory=list)
    confusions: list[tuple[str, Any, int]] = field(default_factory=list)
    charts: list[tuple[str, Any, int]] = field(default_factory=list)
    images: list[tuple[str, Any, int]] = field(default_factory=list)
    systems: list[tuple[Any, ...]] = field(default_factory=list)
    flushes: int = 0
    closed: bool = False
    sleep_per_call: float = 0.0
    fail_every_nth: int = 0
    _calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def log_metric(self, name: str, value: float, step: int) -> None:
        with self._lock:
            self._calls += 1
            should_fail = (
                self.fail_every_nth > 0 and self._calls % self.fail_every_nth == 0
            )
        if self.sleep_per_call:
            time.sleep(self.sleep_per_call)
        if should_fail:
            raise RuntimeError(f"intentional fake-sink failure on call {self._calls}")
        with self._lock:
            self.metrics.append((name, value, step))

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        with self._lock:
            self.examples.append((name, data, step))

    def log_confusion_matrix(self, name: str, cm: Any, step: int) -> None:
        with self._lock:
            self.confusions.append((name, cm, step))

    def log_chart(self, name: str, chart: Any, step: int) -> None:
        with self._lock:
            self.charts.append((name, chart, step))

    def log_image(self, name: str, image: Any, step: int) -> None:
        with self._lock:
            self.images.append((name, image, step))

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        with self._lock:
            self.systems.append(
                (
                    ram_used_gb,
                    ram_total_gb,
                    gpu_mem_used_gb,
                    gpu_mem_total_gb,
                    gpu_util_pct,
                )
            )

    def flush(self) -> None:
        with self._lock:
            self.flushes += 1

    def close(self) -> None:
        with self._lock:
            self.closed = True


class TestAsyncSink:
    def test_log_returns_quickly_when_inner_is_slow(self) -> None:
        inner = _FakeSink(sleep_per_call=0.1)
        sink = AsyncSink(inner)
        try:
            t0 = time.perf_counter()
            for i in range(20):
                sink.log_metric("loss", float(i), step=i)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"log_metric should return immediately, took {elapsed:.3f}s"
            )
        finally:
            sink.close()

    def test_flush_blocks_until_drained(self) -> None:
        inner = _FakeSink(sleep_per_call=0.05)
        sink = AsyncSink(inner)
        try:
            for i in range(10):
                sink.log_metric("loss", float(i), step=i)
            assert len(inner.metrics) < 10
            sink.flush()
            assert len(inner.metrics) == 10
            assert inner.flushes >= 1
        finally:
            sink.close()

    def test_close_drains_pending_items(self) -> None:
        inner = _FakeSink(sleep_per_call=0.01)
        sink = AsyncSink(inner)
        for i in range(50):
            sink.log_metric("loss", float(i), step=i)
        sink.close()
        assert len(inner.metrics) == 50
        assert inner.closed is True

    def test_items_dispatched_in_fifo_order(self) -> None:
        inner = _FakeSink()
        sink = AsyncSink(inner)
        try:
            for i in range(500):
                sink.log_metric("loss", float(i), step=i)
            sink.flush()
            steps = [m[2] for m in inner.metrics]
            assert steps == list(range(500))
        finally:
            sink.close()

    def test_worker_survives_exceptions_in_inner_sink(self) -> None:
        inner = _FakeSink(fail_every_nth=3)
        sink = AsyncSink(inner)
        try:
            for i in range(20):
                sink.log_metric("loss", float(i), step=i)
            sink.flush()
            assert sink._thread.is_alive()
            assert len(inner.metrics) > 10
        finally:
            sink.close()

    def test_high_water_warning_fires_once(self) -> None:
        inner = _FakeSink()
        sink = AsyncSink(inner)
        try:
            with (
                mock.patch.object(
                    sink._queue, "qsize", return_value=_HIGH_WATER_MARK + 1
                ),
                mock.patch("extty.async_sink.logger") as logger_mock,
            ):
                sink.log_metric("loss", 0.0, step=0)
                sink.log_metric("loss", 0.0, step=1)
                sink.log_metric("loss", 0.0, step=2)
                assert logger_mock.warning.call_count == 1
        finally:
            sink.close()

    def test_log_example_dispatched(self) -> None:
        inner = _FakeSink()
        sink = AsyncSink(inner)
        try:
            sink.log_example("val/ex", {"prompt": "hi"}, step=7)
            sink.flush()
            assert inner.examples == [("val/ex", {"prompt": "hi"}, 7)]
        finally:
            sink.close()

    def test_log_image_dispatched(self) -> None:
        from PIL import Image as PILImage

        inner = _FakeSink()
        sink = AsyncSink(inner)
        try:
            image = extty.Image(PILImage.new("RGB", (2, 2), "red"))
            sink.log_image("val/dets", image, step=3)
            sink.flush()
            assert inner.images == [("val/dets", image, 3)]
        finally:
            sink.close()

    def test_log_system_dispatched(self) -> None:
        inner = _FakeSink()
        sink = AsyncSink(inner)
        try:
            sink.log_system(8.0, 32.0, 4.0, 24.0, 50.0)
            sink.flush()
            assert inner.systems == [(8.0, 32.0, 4.0, 24.0, 50.0)]
        finally:
            sink.close()


class TestAtexitSafetyNet:
    def test_atexit_handler_finishes_active_run(self, tmp_path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            extty.init("atexit-test", name="r1", system_metrics=False)
            assert extty._active_run is not None
            run = extty._active_run

            extty._atexit_finish_active_run()

            assert extty._active_run is None
            assert run._finished is True

    def test_atexit_handler_noop_when_no_active_run(self) -> None:
        extty._active_run = None
        extty._atexit_finish_active_run()
        assert extty._active_run is None

    def test_atexit_registered_only_once_across_inits(self, tmp_path) -> None:
        extty._atexit_registered = False
        with (
            mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"),
            mock.patch("extty.atexit.register") as register_mock,
        ):
            extty.init("atexit-once", name="r1", system_metrics=False)
            extty.init("atexit-once", name="r2", system_metrics=False)
            extty.init("atexit-once", name="r3", system_metrics=False)
            extty.finish()
            assert register_mock.call_count == 1
            assert extty._atexit_registered is True
