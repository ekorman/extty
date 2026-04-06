"""extty: Terminal-native ML experiment tracker."""

import copy
import dataclasses
import functools
import warnings
from datetime import datetime

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from extty.example import BatchExample, Example
from extty.compare import compare, config_diff, plot_metric, reduce_metric
from extty.query import RunData, get_run, get_runs
from extty.run import Run
from extty.storage import (
    ExampleRecord,
    MetricPoint,
    SystemMetricPoint,
    log_model_evaluation,
    generate_random_name,
    get_runs_dir,
)
from extty.artifact import (
    ArtifactMeta,
    delete_artifact,
    get_artifact,
    list_artifacts,
    load_artifact,
    save_artifact as _save_artifact_raw,
)
from extty.s3 import S3Config

__all__ = [
    "init",
    "log",
    "log_evaluation",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_from",
    "finish",
    "Run",
    "RunData",
    "get_runs",
    "get_run",
    "push",
    "list_local_runs",
    "compare",
    "config_diff",
    "plot_metric",
    "reduce_metric",
    "Example",
    "BatchExample",
    "MetricPoint",
    "SystemMetricPoint",
    "ExampleRecord",
    "has_active_run",
    "ArtifactMeta",
    "save_artifact",
    "list_artifacts",
    "load_artifact",
    "get_artifact",
    "delete_artifact",
]

# Version is managed by setuptools_scm
try:
    __version__ = version("extty")
except PackageNotFoundError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

_active_run: Run | None = None


def has_active_run() -> bool:
    return _active_run is not None


def init(
    project: str,
    *,
    name: str | None = None,
    config: dict[str, Any] | None = None,
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
    _active_run = Run(
        project,
        name=name,
        config=config,
        system_metrics=system_metrics,
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


def save_checkpoint(
    step: int,
    *,
    path: str | None = None,
    state_dict: Any = None,
    optimizer_state_dict: Any = None,
) -> None:
    """
    Save a checkpoint to S3 for the active run.

    Parameters
    ----------
    step : int
        The training step for this checkpoint.
    path : str or None
        Path to a local file to upload directly.
    state_dict : Any or None
        Model state dict to serialize with torch.save.
    optimizer_state_dict : Any or None
        Optimizer state dict to include when using state_dict.
    """
    if _active_run is None:
        raise RuntimeError("No active run. Call extty.init() first.")
    _active_run.save_checkpoint(
        step,
        path=path,
        state_dict=state_dict,
        optimizer_state_dict=optimizer_state_dict,
    )


def load_checkpoint(
    step: int,
    *,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint for the active run, downloading from S3 if needed.

    Parameters
    ----------
    step : int
        The training step to load.
    load_optimizer : bool, default True
        Whether to include the optimizer state in the result.

    Returns
    -------
    dict[str, Any]
        Contains ``"model_state_dict"`` and optionally
        ``"optimizer_state_dict"``.
    """
    if _active_run is None:
        raise RuntimeError("No active run. Call extty.init() first.")
    return _active_run.load_checkpoint(step, load_optimizer=load_optimizer)


def load_checkpoint_from(
    project: str,
    run_name: str,
    step: int,
    *,
    load_optimizer: bool = True,
    s3_config: S3Config | None = None,
) -> dict[str, Any]:
    """
    Load a checkpoint from any run, downloading from S3 if needed.

    Parameters
    ----------
    project : str
        The project name.
    run_name : str
        The run name.
    step : int
        The training step to load.
    load_optimizer : bool, default True
        Whether to include the optimizer state in the result.
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.

    Returns
    -------
    dict[str, Any]
        Contains ``"model_state_dict"`` and optionally
        ``"optimizer_state_dict"``.
    """
    from extty.s3 import S3Storage

    if s3_config is None:
        s3_config = S3Config.load()
    if s3_config is None:
        raise RuntimeError(
            "S3 storage is not configured. Set EXTTY_S3_BUCKET or provide s3_config."
        )
    storage = S3Storage(s3_config, project, run_name)
    return storage.load_checkpoint(step, load_optimizer=load_optimizer)


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


def save_artifact(
    name: str,
    path: str | Path,
    *,
    description: str = "",
    metadata: dict[str, Any] | None = None,
    s3_config: S3Config | None = None,
) -> ArtifactMeta:
    """
    Save an artifact to S3.

    If an active run exists, the artifact is automatically associated
    with it.

    Parameters
    ----------
    name : str
        Unique name for this artifact.
    path : str or Path
        Local file or directory to upload.
    description : str
        Human-readable description.
    metadata : dict[str, Any], optional
        User-defined metadata (arbitrary JSON-serializable dict).
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.

    Returns
    -------
    ArtifactMeta
        Metadata for the saved artifact.
    """
    run_project = _active_run.project if _active_run is not None else None
    run_name = _active_run.name if _active_run is not None else None
    return _save_artifact_raw(
        name,
        path,
        description=description,
        metadata=metadata,
        s3_config=s3_config,
        run_project=run_project,
        run_name=run_name,
    )


P = ParamSpec("P")
T = TypeVar("T")


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    def _convert(val: Any) -> Any:
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            return dataclasses.asdict(val)
        if isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_convert(v) for v in val]
        return val

    return {k: _convert(v) for k, v in config.items()}


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

                kwargs_copy = _sanitize_config(kwargs_copy)

                run_name = name

                if run_name is None and name_kwarg is not None:
                    run_name = kwargs_copy.pop(name_kwarg)

                init(
                    project=project,
                    name=run_name,
                    config=kwargs_copy,
                    system_metrics=system_metrics,
                )
                return fn(*args, **kwargs)
            finally:
                global _active_run
                if _active_run is not None:
                    finish()

        return wrapper

    return dec


def list_local_runs(project: str | None = None) -> list[tuple[str, str]]:
    """
    List all local runs.

    Parameters
    ----------
    project : str, optional
        Filter to runs in this project only.

    Returns
    -------
    list[tuple[str, str]]
        List of (project, run_name) tuples, sorted by start time
        (most recent first).
    """
    return [(r.project, r.name) for r in get_runs(project=project)]


def push(
    target: str | None = None,
    *,
    s3_config: S3Config | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """
    Push local runs to S3.

    Parameters
    ----------
    target : str, optional
        Target to push. Format: "project/run-name" for single run,
        "project/" for all runs in project, or None for all runs.
    s3_config : S3Config, optional
        S3 configuration. If not provided, reads from environment variables.
    force : bool, default False
        Overwrite remote data without merging.
    dry_run : bool, default False
        Show what would be pushed without actually pushing.

    Returns
    -------
    list[str]
        List of pushed run paths (project/run_name format).
    """
    if s3_config is None:
        s3_config = S3Config.load()
    if s3_config is None:
        raise ValueError(
            "S3 configuration required. Set EXTTY_S3_BUCKET environment variable "
            "or provide s3_config parameter."
        )

    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 operations. Install with: pip install extty[s3]"
        )

    kwargs: dict = {}
    if s3_config.region:
        kwargs["region_name"] = s3_config.region
    if s3_config.access_key_id and s3_config.secret_access_key:
        kwargs["aws_access_key_id"] = s3_config.access_key_id
        kwargs["aws_secret_access_key"] = s3_config.secret_access_key
    if s3_config.endpoint_url:
        kwargs["endpoint_url"] = s3_config.endpoint_url

    client = boto3.client("s3", **kwargs)

    project_filter = None
    run_filter = None
    if target:
        if target.endswith("/"):
            project_filter = target.rstrip("/")
        elif "/" in target:
            project_filter, run_filter = target.split("/", 1)
        else:
            project_filter = target

    runs_to_push = []
    for proj, run_name in list_local_runs(project_filter):
        if run_filter is not None and run_name != run_filter:
            continue
        runs_to_push.append((proj, run_name))

    if dry_run:
        for proj, run_name in runs_to_push:
            print(f"Would push: {proj}/{run_name}")
        return [f"{p}/{r}" for p, r in runs_to_push]

    pushed = []
    runs_dir = get_runs_dir()
    for proj, run_name in runs_to_push:
        local_run_dir = runs_dir / proj / run_name
        s3_prefix = f"{s3_config.prefix}/runs/{proj}/{run_name}"

        _push_run_to_s3(client, s3_config.bucket, s3_prefix, local_run_dir, force=force)
        pushed.append(f"{proj}/{run_name}")
        print(f"Pushed: {proj}/{run_name}")

    return pushed


def _push_run_to_s3(
    client,
    bucket: str,
    s3_prefix: str,
    local_run_dir,
    force: bool = False,
) -> None:
    """Push a single run directory to S3."""
    import json

    meta_path = local_run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_content = f.read()
        if force:
            client.put_object(
                Bucket=bucket,
                Key=f"{s3_prefix}/meta.json",
                Body=meta_content.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            try:
                response = client.get_object(
                    Bucket=bucket, Key=f"{s3_prefix}/meta.json"
                )
                remote_meta = json.loads(response["Body"].read().decode("utf-8"))
                local_meta = json.loads(meta_content)
                merged_meta = _merge_meta(local_meta, remote_meta)
                client.put_object(
                    Bucket=bucket,
                    Key=f"{s3_prefix}/meta.json",
                    Body=json.dumps(merged_meta, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )
            except client.exceptions.NoSuchKey:
                client.put_object(
                    Bucket=bucket,
                    Key=f"{s3_prefix}/meta.json",
                    Body=meta_content.encode("utf-8"),
                    ContentType="application/json",
                )

    metrics_dir = local_run_dir / "metrics"
    if metrics_dir.exists():
        for csv_file in metrics_dir.rglob("*.csv"):
            relative_path = csv_file.relative_to(metrics_dir)
            s3_key = f"{s3_prefix}/metrics/{relative_path}"
            _push_csv_file(client, bucket, s3_key, csv_file, force=force)

    examples_dir = local_run_dir / "examples"
    if examples_dir.exists():
        for jsonl_file in examples_dir.rglob("*.jsonl"):
            relative_path = jsonl_file.relative_to(examples_dir)
            s3_key = f"{s3_prefix}/examples/{relative_path}"
            _push_jsonl_file(client, bucket, s3_key, jsonl_file, force=force)

    system_csv = local_run_dir / "system.csv"
    if system_csv.exists():
        s3_key = f"{s3_prefix}/system.csv"
        _push_csv_file(client, bucket, s3_key, system_csv, force=force)


def _merge_meta(local: dict, remote: dict) -> dict:
    """Merge meta.json from local and remote."""
    if local.get("status") == "completed":
        return local
    if remote.get("status") == "completed":
        return remote
    local_finished = local.get("finished_at")
    remote_finished = remote.get("finished_at")
    if local_finished and remote_finished:
        return local if local_finished >= remote_finished else remote
    if local_finished:
        return local
    if remote_finished:
        return remote
    return local


def _push_csv_file(client, bucket: str, s3_key: str, local_path, force: bool) -> None:
    """Push a CSV file to S3, optionally merging with existing data."""
    import csv
    import io

    with open(local_path) as f:
        reader = csv.DictReader(f)
        local_rows = list(reader)
        if not local_rows:
            return
        fieldnames = reader.fieldnames or []

    if force:
        with open(local_path, "rb") as f:
            client.put_object(
                Bucket=bucket, Key=s3_key, Body=f.read(), ContentType="text/csv"
            )
        return

    existing_keys: set[tuple] = set()
    existing_rows: list[dict] = []
    try:
        response = client.get_object(Bucket=bucket, Key=s3_key)
        content = response["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            key = (row.get("step", ""), row.get("timestamp", ""))
            existing_keys.add(key)
            existing_rows.append(row)
    except client.exceptions.NoSuchKey:
        pass

    for row in local_rows:
        key = (row.get("step", ""), row.get("timestamp", ""))
        if key not in existing_keys:
            existing_keys.add(key)
            existing_rows.append(row)

    existing_rows.sort(
        key=lambda r: (float(r.get("step", 0)), float(r.get("timestamp", 0)))
    )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing_rows)

    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=output.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )


def _push_jsonl_file(client, bucket: str, s3_key: str, local_path, force: bool) -> None:
    """Push a JSONL file to S3, optionally merging with existing data."""
    import json

    with open(local_path) as f:
        local_records = [json.loads(line) for line in f if line.strip()]

    if force:
        with open(local_path, "rb") as f:
            client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=f.read(),
                ContentType="application/x-ndjson",
            )
        return

    existing_keys: set[tuple] = set()
    existing_records: list[dict] = []
    try:
        response = client.get_object(Bucket=bucket, Key=s3_key)
        content = response["Body"].read().decode("utf-8")
        for line in content.strip().split("\n"):
            if line:
                record = json.loads(line)
                key = (record.get("step"), record.get("timestamp"))
                existing_keys.add(key)
                existing_records.append(record)
    except client.exceptions.NoSuchKey:
        pass

    for record in local_records:
        key = (record.get("step"), record.get("timestamp"))
        if key not in existing_keys:
            existing_keys.add(key)
            existing_records.append(record)

    existing_records.sort(key=lambda r: (r.get("step", 0), r.get("timestamp", 0)))

    output = "\n".join(json.dumps(r) for r in existing_records)
    if output:
        output += "\n"

    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=output.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
