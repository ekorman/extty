"""Tests for extty.compare module."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from extty.compare import _ema, compare, config_diff, plot_metric, reduce_metric
from extty.query import RunData
from extty.storage import MetaData, MetricPoint, RunStorage


def make_run(
    tmp_path: Path,
    project: str,
    name: str,
    config: dict[str, Any],
    metrics_data: dict[str, list[tuple[int, float]]] | None = None,
) -> RunData:
    """
    Create a run directory with meta.json and optional metric CSVs,
    then return a RunData pointing at it.
    """
    run_dir = tmp_path / "runs" / project / name
    storage = RunStorage(run_dir=run_dir)
    meta = MetaData(
        project=project,
        run_name=name,
        config=config,
        started_at="2024-01-01T00:00:00Z",
        finished_at="2024-01-01T01:00:00Z",
        status="completed",
    )
    storage.write_meta(meta)

    for metric_name, points in (metrics_data or {}).items():
        for step, value in points:
            storage.log_metric(metric_name, value, step)
    storage.flush()

    return RunData(
        project=project,
        name=name,
        config=config,
        started_at=meta.started_at,
        finished_at=meta.finished_at,
        status=meta.status,
        _storage=storage,
    )


class TestReduceMetric:
    def _pts(self, values: list[float]) -> list[MetricPoint]:
        return [
            MetricPoint(step=i, timestamp=float(i), value=v)
            for i, v in enumerate(values)
        ]

    def test_last(self) -> None:
        assert reduce_metric(self._pts([1.0, 2.0, 3.0]), "last") == 3.0

    def test_min(self) -> None:
        assert reduce_metric(self._pts([3.0, 1.0, 2.0]), "min") == 1.0

    def test_max(self) -> None:
        assert reduce_metric(self._pts([1.0, 3.0, 2.0]), "max") == 3.0

    def test_mean(self) -> None:
        assert reduce_metric(self._pts([1.0, 2.0, 3.0]), "mean") == 2.0

    def test_mean_n(self) -> None:
        result = reduce_metric(self._pts([10.0, 1.0, 2.0, 3.0]), "mean:2")
        assert result == 2.5

    def test_ema_n(self) -> None:
        result = reduce_metric(self._pts([1.0, 1.0, 1.0]), "ema:3")
        assert result == pytest.approx(1.0)

    def test_callable(self) -> None:
        custom = lambda pts: pts[0].value + pts[-1].value  # noqa: E731
        assert reduce_metric(self._pts([1.0, 2.0, 3.0]), custom) == 4.0

    def test_empty_returns_nan(self) -> None:
        import math

        assert math.isnan(reduce_metric([], "last"))

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reduction"):
            reduce_metric(self._pts([1.0]), "bogus")

    def test_ema_without_span_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a span"):
            reduce_metric(self._pts([1.0]), "ema")

    def test_default_reduction_is_last(self) -> None:
        assert reduce_metric(self._pts([1.0, 5.0])) == 5.0


class TestConfigDiff:
    def test_finds_differing_keys(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001, "batch": 32})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.01, "batch": 32})
        diff = config_diff([r1, r2])
        assert "lr" in diff
        assert "batch" not in diff
        assert diff["lr"] == [0.001, 0.01]

    def test_excludes_private_by_default(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001, "_git_hash": "aaa"})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.001, "_git_hash": "bbb"})
        diff = config_diff([r1, r2])
        assert "_git_hash" not in diff

    def test_includes_private_when_requested(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"_git_hash": "aaa"})
        r2 = make_run(tmp_path, "p", "r2", {"_git_hash": "bbb"})
        diff = config_diff([r1, r2], include_private=True)
        assert "_git_hash" in diff

    def test_empty_when_all_same(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.001})
        assert config_diff([r1, r2]) == {}

    def test_missing_keys_treated_as_different(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.001, "wd": 0.01})
        diff = config_diff([r1, r2])
        assert "wd" in diff
        assert diff["wd"] == [None, 0.01]

    def test_single_run_returns_empty(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001})
        assert config_diff([r1]) == {}


pd = pytest.importorskip("pandas")


class TestCompare:
    def test_basic_table_shape(self, tmp_path: Path) -> None:
        r1 = make_run(
            tmp_path, "p", "r1", {"lr": 0.001}, {"loss": [(0, 0.5), (1, 0.3)]}
        )
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.01}, {"loss": [(0, 0.6), (1, 0.4)]})
        df = compare([r1, r2])
        assert len(df) == 2
        assert "name" in df.columns
        assert "lr" in df.columns
        assert "loss" in df.columns

    def test_all_config_flag(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001, "batch": 32})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.001, "batch": 32})
        df = compare([r1, r2], metrics={}, all_config=True)
        assert "lr" in df.columns
        assert "batch" in df.columns

    def test_differing_config_only_by_default(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001, "batch": 32})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.01, "batch": 32})
        df = compare([r1, r2], metrics={})
        assert "lr" in df.columns
        assert "batch" not in df.columns

    def test_custom_reductions(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 1.0), (1, 0.5), (2, 0.2)]})
        df = compare([r1], metrics={"loss": "min"})
        assert df["loss"].iloc[0] == pytest.approx(0.2)

    def test_missing_metric_is_nan(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)]})
        r2 = make_run(tmp_path, "p", "r2", {}, {})
        df = compare([r1, r2], metrics={"loss": "last"})
        assert pd.isna(df["loss"].iloc[1])

    def test_project_column_when_varies(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "a", "r1", {})
        r2 = make_run(tmp_path, "b", "r2", {})
        df = compare([r1, r2], metrics={})
        assert "project" in df.columns

    def test_no_project_column_when_same(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {})
        r2 = make_run(tmp_path, "p", "r2", {})
        df = compare([r1, r2], metrics={})
        assert "project" not in df.columns

    def test_column_ordering(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "a", "r1", {"z": 1, "a": 2}, {"loss": [(0, 0.5)]})
        r2 = make_run(tmp_path, "b", "r2", {"z": 2, "a": 3}, {"loss": [(0, 0.6)]})
        df = compare([r1, r2], metrics={"loss": "last"})
        cols = list(df.columns)
        assert cols[0] == "project"
        assert cols[1] == "name"
        a_idx = cols.index("a")
        z_idx = cols.index("z")
        loss_idx = cols.index("loss")
        assert a_idx < z_idx < loss_idx

    def test_private_key_filtering(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001, "_git": "abc"})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.01, "_git": "def"})
        df = compare([r1, r2], metrics={})
        assert "_git" not in df.columns

    def test_auto_discover_metrics(self, tmp_path: Path) -> None:
        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)], "acc": [(0, 0.9)]})
        df = compare([r1])
        assert "loss" in df.columns
        assert "acc" in df.columns


mpl = pytest.importorskip("matplotlib")


class TestPlotMetric:
    def test_returns_figure(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5), (1, 0.3)]})
        fig = plot_metric([r1], "loss")
        assert isinstance(fig, mpl.figure.Figure)
        plt.close(fig)

    def test_correct_line_count(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5), (1, 0.3)]})
        r2 = make_run(tmp_path, "p", "r2", {}, {"loss": [(0, 0.6), (1, 0.4)]})
        fig = plot_metric([r1, r2], "loss")
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_smoothing_doubles_lines(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5), (1, 0.3)]})
        fig = plot_metric([r1], "loss", smooth=3)
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_missing_metric_skipped_with_warning(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_metric([r1], "nonexistent")
            assert any("missing metric" in str(warning.message) for warning in w)
        plt.close(fig)

    def test_legend_name(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)]})
        fig = plot_metric([r1], "loss", legend="name")
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["r1"]
        plt.close(fig)

    def test_legend_full(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)]})
        fig = plot_metric([r1], "loss", legend="full")
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["p/r1"]
        plt.close(fig)

    def test_legend_diff(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {"lr": 0.001}, {"loss": [(0, 0.5)]})
        r2 = make_run(tmp_path, "p", "r2", {"lr": 0.01}, {"loss": [(0, 0.6)]})
        fig = plot_metric([r1, r2], "loss", legend="diff")
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "lr=0.001" in labels[0]
        assert "lr=0.01" in labels[1]
        plt.close(fig)

    def test_custom_ax(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)]})
        fig, ax = plt.subplots()
        returned_fig = plot_metric([r1], "loss", ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_title_and_labels(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        r1 = make_run(tmp_path, "p", "r1", {}, {"loss": [(0, 0.5)]})
        fig = plot_metric(
            [r1], "loss", title="My Title", xlabel="iteration", ylabel="L"
        )
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "iteration"
        assert ax.get_ylabel() == "L"
        plt.close(fig)


class TestEma:
    def test_single_value(self) -> None:
        assert _ema([5.0], 3) == [5.0]

    def test_constant_series(self) -> None:
        result = _ema([2.0, 2.0, 2.0, 2.0], 3)
        assert all(v == pytest.approx(2.0) for v in result)

    def test_smoothing_reduces_variance(self) -> None:
        noisy = [1.0, 10.0, 1.0, 10.0, 1.0, 10.0]
        smoothed = _ema(noisy, 3)
        raw_var = sum((v - 5.5) ** 2 for v in noisy) / len(noisy)
        smooth_var = sum(
            (v - sum(smoothed) / len(smoothed)) ** 2 for v in smoothed
        ) / len(smoothed)
        assert smooth_var < raw_var

    def test_empty(self) -> None:
        assert _ema([], 3) == []
