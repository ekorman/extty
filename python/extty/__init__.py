"""extty: Terminal-native ML experiment tracker."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from extty.run import Run, ServerConfig
from extty.server import ServerSettings

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
    config: dict[str, Any] | None = None,
    system_metrics: bool = True,
    server: bool = False,
    server_host: str = "0.0.0.0",
    server_port: int = 0,
    server_token: str | None = None,
    create_modal_tunnel: bool = False,
    server_max_metric_points: int = 10_000,
    server_max_example_points: int = 5_000,
    server_max_system_points: int = 2_000,
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
    server : bool, default False
        Whether to start an in-memory HTTP server for remote TUI access.
    server_host : str, default "0.0.0.0"
        Host interface for the server.
    server_port : int, default 0
        Port for the server (0 chooses a random available port).
    server_token : str, optional
        Token for Authorization header; auto-generated if omitted.
    server_max_metric_points : int, default 10000
        Maximum points per metric series in memory.
    server_max_example_points : int, default 5000
        Maximum examples per series in memory.
    server_max_system_points : int, default 2000
        Maximum system metric samples in memory.

    Returns
    -------
    Run
        The initialized run object.
    """
    global _active_run
    if _active_run is not None:
        _active_run.finish()
    server_config = None
    if server:
        settings = ServerSettings(
            host=server_host,
            port=server_port,
            token=server_token,
            max_metric_points=server_max_metric_points,
            max_example_points=server_max_example_points,
            max_system_points=server_max_system_points,
            create_modal_tunnel=create_modal_tunnel,
        )
        server_config = ServerConfig(enabled=True, settings=settings)
    _active_run = Run(
        project,
        name=name,
        config=config,
        system_metrics=system_metrics,
        server=server_config,
    )
    return _active_run


def log(metrics: dict[str, float | dict[str, Any]], *, step: int) -> None:
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
