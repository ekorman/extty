"""extty: Terminal-native ML experiment tracker."""

import copy
import functools
import warnings
from datetime import datetime

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, ParamSpec, TypeVar

from extty.example import BatchExample, Example
from extty.run import Run, ServerConfig
from extty.server import ServerSettings
from extty.storage import log_model_evaluation, generate_random_name

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
    project: str,
    model: str,
    *,
    name: str | None = None,
    metrics: dict[str, float] | None = None,
    examples: list[Example] | None = None,
    model_config: dict[str, Any] | None = None,
    eval_config: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> None:
    """
    Log an evaluation for a model.

    Parameters
    ----------
    project : str
        Name of the project.
    model : str
        Name of the model.
    name : str
        Name of this evaluation (e.g., "gsm8k", "humaneval").
    metrics : dict[str, float], optional
        Evaluation metrics (e.g., {"accuracy": 0.85}).
    examples : list[Example], optional
        Sample outputs as Example objects.
    model_config : dict[str, Any], optional
        Model configuration (stored on model's meta.json).
    eval_config : dict[str, Any], optional
        Evaluation configuration (stored with evaluation).
    started_at : str, optional
        ISO timestamp when the evaluation started.
    finished_at : str, optional
        ISO timestamp when the evaluation finished.
    """
    name = name or generate_random_name()
    examples_dicts = [ex.to_dict() for ex in examples] if examples else None
    log_model_evaluation(
        project=project,
        model=model,
        name=name,
        metrics=metrics,
        examples=examples_dicts,
        model_config=model_config,
        eval_config=eval_config,
        started_at=started_at,
        finished_at=finished_at,
    )


def finish() -> None:
    """Finish the current run and flush all data."""
    global _active_run
    if _active_run is None:
        raise RuntimeError("No active run to finish.")
    _active_run.finish()
    _active_run = None


P = ParamSpec("P")
T = TypeVar("T")


def evaluation(
    project: str,
    *,
    name: str | None = None,
    name_kwarg: str | None = None,
    model: str | None = None,
    model_kwarg: str | None = None,
    model_config_kwargs: list[str],
    eval_config_kwargs: list[str],
):
    def dec(
        fn: Callable[P, tuple[dict[str, float], list[Example]]],
    ) -> Callable[P, tuple[dict[str, float], list[Example]]]:
        @functools.wraps(fn)
        def wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> tuple[dict[str, float], list[Example]]:
            kwargs_copy = copy.deepcopy(kwargs)

            eval_name = name
            if eval_name is None and name_kwarg is not None:
                eval_name = kwargs_copy.pop(name_kwarg)

            model_name = model
            if model_name is None and model_kwarg is not None:
                model_name = kwargs_copy.pop(model_kwarg)

            model_config = {k: kwargs_copy[k] for k in model_config_kwargs}
            eval_config = {k: kwargs_copy[k] for k in eval_config_kwargs}

            started_at = datetime.now().isoformat()
            metrics, examples = fn(*args, **kwargs)
            finished_at = datetime.now().isoformat()
            log_evaluation(
                project=project,
                model=model_name,
                name=eval_name,
                model_config=model_config,
                eval_config=eval_config,
                metrics=metrics,
                examples=examples,
                started_at=started_at,
                finished_at=finished_at,
            )
            return metrics, examples

        return wrapper

    return dec


def experiment(
    project: str,
    name: str | None = None,
    name_kwarg: str | None = None,
    conf_kwargs: list[str] | None = None,
    non_conf_kwargs: list[str] | None = None,
    system_metrics: bool = True,
    server: bool | Callable[..., bool] = False,
    server_host: str = "0.0.0.0",
    server_port: int = 0,
    server_token: str | None = None,
    server_max_metric_points: int = 10_000,
    server_max_example_points: int = 5_000,
    server_max_system_points: int = 2_000,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if conf_kwargs is not None and non_conf_kwargs is not None:
        raise ValueError(
            "cannot pass values for both `conf_kwargs` and `non_conf_kwargs`"
        )

    def dec(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if len(args) > 0:
                warnings.warn(
                    f"non-keyword args passed to {fn} will not be logged to `extty`."
                )
            try:
                kwargs_copy = copy.deepcopy(kwargs)
                if conf_kwargs is not None:
                    kwargs_copy = {k: kwargs_copy[k] for k in conf_kwargs}
                if non_conf_kwargs is not None:
                    kwargs_copy = {
                        k: v for k, v in kwargs_copy.items() if k not in non_conf_kwargs
                    }

                run_name = name

                if run_name is None and name_kwarg is not None:
                    run_name = kwargs_copy.pop(name_kwarg)

                if callable(server):
                    run_server = server()
                else:
                    run_server = server
                init(
                    project=project,
                    name=run_name,
                    config=kwargs_copy,
                    system_metrics=system_metrics,
                    server=run_server,
                    server_host=server_host,
                    server_port=server_port,
                    server_token=server_token,
                    server_max_metric_points=server_max_metric_points,
                    server_max_example_points=server_max_example_points,
                    server_max_system_points=server_max_system_points,
                )
                return fn(*args, **kwargs)
            finally:
                global _active_run
                if _active_run is not None:
                    finish()

        return wrapper

    return dec
