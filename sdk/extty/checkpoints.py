"""
On-disk checkpoint layout shared by the local and S3 backends.

Every checkpoint is written to ``<run dir>/checkpoints/<step>/`` first; S3,
when configured, receives an upload of that directory. Each file is written
atomically, and the per-step ``meta.json`` is written last, so its presence
marks a complete local copy.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from extty import storage

META_FILE = "meta.json"
LEGACY_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    """Directory holding a run's checkpoint for ``step``."""
    return run_dir / "checkpoints" / str(step)


def local_checkpoint_dir(project: str, run_name: str, step: int) -> Path:
    """Directory holding the checkpoint for ``step`` of a run in the runs dir."""
    project_dir = project if project else "_default"
    return checkpoint_dir(storage.get_runs_dir() / project_dir / run_name, step)


def file_names(meta: dict[str, Any]) -> list[str]:
    """
    Names of the files recorded in a checkpoint's meta entry.

    Parameters
    ----------
    meta : dict[str, Any]
        A ``checkpoints.json`` / ``meta.json`` entry. Older entries list bare
        names instead of ``{"name", "size_bytes"}`` dicts, and the oldest have
        no ``files`` at all.

    Returns
    -------
    list[str]
        The recorded names, or ``["checkpoint.pt"]`` when none are recorded.
    """
    names: list[str] = []
    for entry in meta.get("files", []):
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            names.append(entry["name"])
    return names or [LEGACY_FILE]


def required_files(names: list[str], *, load_optimizer: bool) -> list[str]:
    """The subset of ``names`` needed to load a checkpoint."""
    return [name for name in names if load_optimizer or name != OPTIMIZER_FILE]


def _write_atomic(target: Path, write: Callable[[Path], object]) -> int:
    tmp = target.with_name(target.name + ".part")
    write(tmp)
    tmp.replace(target)
    return target.stat().st_size


def write_meta(dest: Path, meta: dict[str, Any]) -> None:
    """Write ``meta`` as the ``meta.json`` of the checkpoint in ``dest``."""
    _write_atomic(
        dest / META_FILE, lambda tmp: tmp.write_text(json.dumps(meta, indent=2))
    )


def read_meta(dest: Path) -> dict[str, Any] | None:
    """The ``meta.json`` of the checkpoint in ``dest``, or None if absent or corrupt."""
    try:
        return json.loads((dest / META_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_checkpoint(
    dest: Path,
    step: int,
    *,
    path: str | None = None,
    state_dict: Any = None,
    optimizer_state_dict: Any = None,
) -> dict[str, Any]:
    """
    Write a checkpoint into ``dest``.

    Parameters
    ----------
    dest : Path
        Checkpoint directory; created if missing.
    step : int
        The training step for this checkpoint.
    path : str or None
        Path to an existing file, copied in as ``checkpoint.pt``.
    state_dict : Any or None
        Model state dict, serialized with ``torch.save`` as ``model.pt``.
    optimizer_state_dict : Any or None
        Optimizer state dict, serialized as ``optimizer.pt`` when using
        ``state_dict``.

    Returns
    -------
    dict[str, Any]
        The ``checkpoints.json`` index entry (``step``, ``timestamp``,
        ``files``), also written to ``dest / "meta.json"``.

    Raises
    ------
    ValueError
        If neither or both of ``path`` and ``state_dict`` are provided.
    """
    if (path is None) == (state_dict is None):
        raise ValueError("Exactly one of `path` or `state_dict` must be provided.")

    writers: dict[str, Callable[[Path], object]]
    if path is not None:
        writers = {LEGACY_FILE: lambda tmp: shutil.copyfile(path, tmp)}
    else:
        import torch

        writers = {MODEL_FILE: lambda tmp: torch.save(state_dict, tmp)}
        if optimizer_state_dict is not None:
            writers[OPTIMIZER_FILE] = lambda tmp: torch.save(optimizer_state_dict, tmp)

    dest.mkdir(parents=True, exist_ok=True)
    (dest / META_FILE).unlink(missing_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    entry: dict[str, Any] = {
        "step": step,
        "timestamp": timestamp,
        "files": [
            {"name": name, "size_bytes": _write_atomic(dest / name, write)}
            for name, write in writers.items()
        ],
    }
    write_meta(dest, entry)
    return entry


def read_checkpoint(
    dest: Path,
    names: list[str],
    *,
    load_optimizer: bool = True,
    map_location: Any = None,
) -> dict[str, Any]:
    """
    Deserialize the checkpoint files in ``dest``.

    Handles both the legacy format (a single ``checkpoint.pt``, optionally
    holding ``model_state_dict`` / ``optimizer_state_dict`` keys) and the
    current one (separate ``model.pt`` / ``optimizer.pt``).

    Parameters
    ----------
    dest : Path
        Checkpoint directory.
    names : list[str]
        The checkpoint's file names, as returned by :func:`file_names`.
    load_optimizer : bool, default True
        Whether to include the optimizer state in the result.
    map_location : Any, optional
        Passed to ``torch.load``; defaults to CPU.

    Returns
    -------
    dict[str, Any]
        Always contains ``"model_state_dict"``. Contains
        ``"optimizer_state_dict"`` when available and ``load_optimizer`` is
        True.
    """
    import torch

    map_location = map_location or torch.device("cpu")

    if names == [LEGACY_FILE]:
        data = torch.load(
            dest / LEGACY_FILE, weights_only=False, map_location=map_location
        )
        result: dict[str, Any] = {
            "model_state_dict": data.get("model_state_dict", data)
        }
        if load_optimizer and "optimizer_state_dict" in data:
            result["optimizer_state_dict"] = data["optimizer_state_dict"]
        return result

    result = {
        "model_state_dict": torch.load(
            dest / MODEL_FILE, weights_only=False, map_location=map_location
        )
    }
    if load_optimizer and OPTIMIZER_FILE in names and (dest / OPTIMIZER_FILE).exists():
        result["optimizer_state_dict"] = torch.load(
            dest / OPTIMIZER_FILE, weights_only=False, map_location=map_location
        )
    return result


def read_local_checkpoint(
    dest: Path,
    *,
    load_optimizer: bool = True,
    map_location: Any = None,
) -> dict[str, Any] | None:
    """
    Load the checkpoint in ``dest`` if a complete local copy is there.

    Parameters
    ----------
    dest : Path
        Checkpoint directory.
    load_optimizer : bool, default True
        Whether to include the optimizer state in the result.
    map_location : Any, optional
        Passed to ``torch.load``; defaults to CPU.

    Returns
    -------
    dict[str, Any] or None
        As :func:`read_checkpoint`, or None if ``dest`` has no ``meta.json``
        or is missing a file needed for this load.
    """
    meta = read_meta(dest)
    if meta is None:
        return None
    names = file_names(meta)
    needed = required_files(names, load_optimizer=load_optimizer)
    if not all((dest / name).exists() for name in needed):
        return None
    return read_checkpoint(
        dest, names, load_optimizer=load_optimizer, map_location=map_location
    )
