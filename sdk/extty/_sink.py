"""Storage sink protocol shared between ``Run`` and the async wrapper."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from extty.chart import Chart
from extty.confusion import ConfusionMatrix


@runtime_checkable
class StorageSink(Protocol):
    def log_metric(self, name: str, value: float, step: int) -> None: ...

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None: ...

    def log_confusion_matrix(
        self, name: str, cm: ConfusionMatrix, step: int
    ) -> None: ...

    def log_chart(self, name: str, chart: Chart, step: int) -> None: ...

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
