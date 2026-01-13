"""extty: Terminal-native ML experiment tracker."""

from importlib.metadata import PackageNotFoundError, version

from extty.run import Run

__all__ = ["init", "log", "finish", "Run", "push", "list_local_runs"]

# Version is managed by setuptools_scm
try:
    __version__ = version("extty")
except PackageNotFoundError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

_active_run: Run | None = None


def init(
    project: str,
    *,
    name: str | None = None,
    config: dict | None = None,
    system_metrics: bool = True,
) -> Run:
    """
    Initialize a new experiment run.

    Parameters
    ----------
    project : str
        Name of the project/experiment group.
    name : str, optional
        Name for this specific run. Auto-generated if not provided.
    config : dict, optional
        Hyperparameters and configuration to log.
    system_metrics : bool, default True
        Whether to automatically collect system metrics (RAM, GPU).

    Returns
    -------
    Run
        The initialized run object.
    """
    global _active_run
    if _active_run is not None:
        _active_run.finish()
    _active_run = Run(project, name=name, config=config, system_metrics=system_metrics)
    return _active_run


def log(metrics: dict[str, float | dict], *, step: int) -> None:
    """
    Log metrics or structured examples for the current step.

    Parameters
    ----------
    metrics : dict[str, float | dict]
        Dictionary of metric names to values or structured example payloads.
    step : int
        The current training step.
    """
    if _active_run is None:
        raise RuntimeError("No active run. Call extty.init() first.")
    _active_run.log(metrics, step=step)


def finish() -> None:
    """Finish the current run and flush all data."""
    global _active_run
    if _active_run is None:
        raise RuntimeError("No active run to finish.")
    _active_run.finish()
    _active_run = None
