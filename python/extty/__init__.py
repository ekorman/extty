"""extty: Terminal-native ML experiment tracker."""

import functools
import warnings

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, ParamSpec, TypeVar

from extty.example import BatchExample, Example
from extty.run import Run, ServerConfig
from extty.server import ServerSettings

__all__ = [
    "init",
    "log",
    "log_evaluation",
    "finish",
    "Run",
    "push",
    "list_local_runs",
    "Example",
    "BatchExample",
]

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
        )
        server_config = ServerConfig(enabled=True, settings=settings)
    _active_run = Run(
        project,
        name=name,
        config=config,
        system_metrics=system_metrics,
        server=server_config,
    )
    print(
        f"extty initialized with run {project}/{_active_run.name}, writing to {_active_run.run_dir}"
    )
    return _active_run


def log(metrics: dict[str, Any], *, step: int) -> None:
    """
    Log metrics or structured examples for the current step.

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary of metric names to values. Values can be:
        - float/int: logged as metric
        - Example: single prompt with grouped responses
        - BatchExample: batch of prompts with grouped responses
    step : int
        The current training step.
    """
    if _active_run is None:
        raise RuntimeError("No active run. Call extty.init() first.")
    _active_run.log(metrics, step=step)


def log_evaluation(
    name: str,
    *,
    metrics: dict[str, float] | None = None,
    examples: list[dict[str, str]] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Log an evaluation snapshot for the current run.

    Parameters
    ----------
    name : str
        Name of this evaluation (e.g., "gsm8k", "humaneval").
    metrics : dict[str, float], optional
        Evaluation metrics (e.g., {"reward_mean": 0.85, "reward_std": 0.12}).
    examples : list[dict[str, str]], optional
        Sample outputs: [{"prompt": str, "response": str}, ...].
    config : dict[str, Any], optional
        Evaluation configuration (e.g., dataset, temperature).
    """
    if _active_run is None:
        raise RuntimeError("No active run. Call extty.init() first.")
    _active_run.log_evaluation(name, metrics=metrics, examples=examples, config=config)


def finish() -> None:
    """Finish the current run and flush all data."""
    global _active_run
    if _active_run is None:
        raise RuntimeError("No active run to finish.")
    _active_run.finish()
    _active_run = None


P = ParamSpec("P")
T = TypeVar("T")


def experiment(
    project: str,
    name: str | None = None,
    system_metrics: bool = True,
    server: bool = False,
    server_host: str = "0.0.0.0",
    server_port: int = 0,
    server_token: str | None = None,
    server_max_metric_points: int = 10_000,
    server_max_example_points: int = 5_000,
    server_max_system_points: int = 2_000,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def dec(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if len(args) > 0:
                warnings.warn(
                    f"non-keyword args passed to {fn} will not be logged to `extty`."
                )
            try:
                init(
                    project=project,
                    name=name,
                    config=kwargs,
                    system_metrics=system_metrics,
                    server=server,
                    server_host=server_host,
                    server_port=server_port,
                    server_token=server_token,
                    server_max_metric_points=server_max_metric_points,
                    server_max_example_points=server_max_example_points,
                    server_max_system_points=server_max_system_points,
                )
                return fn(*args, **kwargs)
            finally:
                finish()

        return wrapper

    return dec
