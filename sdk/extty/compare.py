"""Comparison and visualization utilities for hyperparameter selection."""

from __future__ import annotations

import warnings
from typing import Any, Callable

from extty.query import RunData
from extty.storage import MetricPoint


def _require_pandas():
    try:
        import pandas as pd

        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for this function. "
            "Install with: pip install extty[run-analysis]"
        )


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for this function. "
            "Install with: pip install extty[run-analysis]"
        )


def _ema(values: list[float], span: int) -> list[float]:
    """
    Compute exponential moving average.

    Parameters
    ----------
    values : list[float]
        Input values.
    span : int
        EMA span (window size).

    Returns
    -------
    list[float]
        EMA-smoothed values, same length as input.
    """
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _parse_reduction(spec: str) -> Callable[[list[MetricPoint]], float]:
    """Parse a reduction spec string into a callable."""
    if spec == "last":
        return lambda pts: pts[-1].value
    if spec == "min":
        return lambda pts: min(p.value for p in pts)
    if spec == "max":
        return lambda pts: max(p.value for p in pts)
    if spec == "mean":
        return lambda pts: sum(p.value for p in pts) / len(pts)

    if ":" in spec:
        name, n_str = spec.split(":", 1)
        try:
            n = int(n_str)
        except ValueError:
            raise ValueError(
                f"Invalid span in reduction '{spec}': expected integer after ':'"
            )
        if n <= 0:
            raise ValueError(f"Span must be positive in reduction '{spec}', got {n}")

        if name == "mean":
            return lambda pts, _n=n: sum(p.value for p in pts[-_n:]) / min(len(pts), _n)
        if name == "ema":
            return lambda pts, _n=n: _ema([p.value for p in pts], _n)[-1]
        raise ValueError(f"Unknown reduction '{name}' in '{spec}'")

    if spec in ("ema", "mean:"):
        raise ValueError(
            f"Reduction '{spec}' requires a span, e.g. '{spec.rstrip(':')}:N'"
        )

    raise ValueError(
        f"Unknown reduction '{spec}'. "
        "Expected one of: 'last', 'min', 'max', 'mean', 'mean:N', 'ema:N', or a callable."
    )


def reduce_metric(
    points: list[MetricPoint],
    reduction: str | Callable[[list[MetricPoint]], float] = "last",
) -> float:
    """
    Reduce a metric time series to a scalar value.

    Parameters
    ----------
    points : list[MetricPoint]
        Metric data points.
    reduction : str or callable
        Reduction strategy:
        - ``"last"`` / ``"min"`` / ``"max"`` / ``"mean"`` — operate on all values
        - ``"mean:N"`` — mean of last N points
        - ``"ema:N"`` — EMA with span N, return final value
        - ``Callable[[list[MetricPoint]], float]`` — custom reducer

    Returns
    -------
    float
        Reduced scalar value, or ``float('nan')`` for empty input.

    Raises
    ------
    ValueError
        For unknown reduction strings or missing span.
    """
    if not points:
        return float("nan")

    if callable(reduction) and not isinstance(reduction, str):
        return reduction(points)

    fn = _parse_reduction(reduction)
    return fn(points)


def config_diff(
    runs: list[RunData],
    *,
    include_private: bool = False,
) -> dict[str, list[Any]]:
    """
    Identify config keys whose values differ across runs.

    Parameters
    ----------
    runs : list[RunData]
        Runs to compare.
    include_private : bool, default False
        Include keys prefixed with ``_`` (auto-captured metadata).

    Returns
    -------
    dict[str, list[Any]]
        Mapping from differing config key to list of values (one per run,
        in input order). Missing keys use ``None``.
    """
    if len(runs) <= 1:
        return {}

    all_keys: set[str] = set()
    for run in runs:
        all_keys.update(run.config.keys())

    if not include_private:
        all_keys = {k for k in all_keys if not k.startswith("_")}

    diff: dict[str, list[Any]] = {}
    for key in sorted(all_keys):
        values = [run.config.get(key) for run in runs]
        if len(set(repr(v) for v in values)) > 1:
            diff[key] = values

    return diff


def compare(
    runs: list[RunData],
    metrics: dict[str, str | Callable[[list[MetricPoint]], float]] | None = None,
    *,
    all_config: bool = False,
    include_private: bool = False,
) -> Any:
    """
    Build a DataFrame comparing runs across config and metrics.

    Parameters
    ----------
    runs : list[RunData]
        Runs to compare.
    metrics : dict[str, str | Callable], optional
        Mapping of metric name to reduction (string or callable).
        ``None`` auto-discovers all metrics with ``"last"`` reduction.
    all_config : bool, default False
        Show all config keys. When False, only keys that differ are shown.
    include_private : bool, default False
        Include ``_``-prefixed config keys.

    Returns
    -------
    pd.DataFrame
        Rows = runs, columns = [name] + config keys + metric columns.
        A ``project`` column is included only when project varies across runs.
    """
    pd = _require_pandas()

    if metrics is None:
        all_metric_names: set[str] = set()
        for run in runs:
            all_metric_names.update(run.metric_names)
        metrics = {name: "last" for name in sorted(all_metric_names)}

    projects = {run.project for run in runs}
    include_project = len(projects) > 1

    if all_config:
        config_keys: set[str] = set()
        for run in runs:
            config_keys.update(run.config.keys())
        if not include_private:
            config_keys = {k for k in config_keys if not k.startswith("_")}
        config_columns = sorted(config_keys)
    else:
        diff = config_diff(runs, include_private=include_private)
        config_columns = list(diff.keys())

    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {}
        if include_project:
            row["project"] = run.project
        row["name"] = run.name

        for key in config_columns:
            row[key] = run.config.get(key)

        for metric_name, reduction in metrics.items():
            try:
                points = run.metric(metric_name)
                row[metric_name] = reduce_metric(points, reduction)
            except FileNotFoundError:
                row[metric_name] = float("nan")

        rows.append(row)

    columns: list[str] = []
    if include_project:
        columns.append("project")
    columns.append("name")
    columns.extend(config_columns)
    columns.extend(metrics.keys())

    return pd.DataFrame(rows, columns=columns)


def plot_metric(
    runs: list[RunData],
    metric: str,
    *,
    smooth: int | None = None,
    title: str | None = None,
    xlabel: str = "step",
    ylabel: str | None = None,
    legend: str = "name",
    ax: Any = None,
) -> Any:
    """
    Plot one metric across multiple runs.

    Parameters
    ----------
    runs : list[RunData]
        Runs to plot.
    metric : str
        Metric name to plot.
    smooth : int, optional
        EMA span for smoothing. When set, raw data is shown at alpha=0.2.
    title : str, optional
        Plot title. Defaults to the metric name.
    xlabel : str, default "step"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Defaults to the metric name.
    legend : str, default "name"
        Legend format: ``"name"`` (run name), ``"full"`` (project/name),
        ``"diff"`` (name + differing config values).
    ax : matplotlib Axes, optional
        Axes to plot onto. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    _, plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    diff = config_diff(runs) if legend == "diff" else {}

    for i, run in enumerate(runs):
        try:
            points = run.metric(metric)
        except FileNotFoundError:
            warnings.warn(f"Run '{run.name}' is missing metric '{metric}', skipping.")
            continue

        steps = [p.step for p in points]
        values = [p.value for p in points]

        label = _build_legend_label(run, i, legend, diff)

        if smooth is not None:
            smoothed = _ema(values, smooth)
            ax.plot(steps, values, alpha=0.2)
            ax.plot(steps, smoothed, label=label)
        else:
            ax.plot(steps, values, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.legend()

    return fig


def _build_legend_label(
    run: RunData,
    index: int,
    legend: str,
    diff: dict[str, list[Any]],
) -> str:
    if legend == "full":
        return f"{run.project}/{run.name}"
    if legend == "diff":
        parts = [run.name]
        diff_parts = []
        for key, values in diff.items():
            diff_parts.append(f"{key}={values[index]}")
        if diff_parts:
            parts.append(f"({', '.join(diff_parts)})")
        return " ".join(parts)
    return run.name
