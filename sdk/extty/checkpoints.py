"""
On-disk checkpoint layout shared by the local and S3 backends.

A checkpoint for ``step`` lives in ``<run dir>/checkpoints/<step>/``. Saving
has two steps: :func:`stage_checkpoint` serializes state dicts into that
directory (a ``path`` is left where it is), then S3 receives the staged files
and/or :func:`commit_checkpoint` completes the local copy. Files are written
atomically and ``meta.json`` last, so its presence marks a complete local copy.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from extty.storage import get_run_dir

META_FILE = "meta.json"
LEGACY_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    """Directory holding a run's checkpoint for ``step``."""
    return run_dir / "checkpoints" / str(step)


def local_checkpoint_dir(project: str, run_name: str, step: int) -> Path:
    """Directory holding the checkpoint for ``step`` of a run in the runs dir."""
    return checkpoint_dir(get_run_dir(project, run_name), step)


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


@contextmanager
def _atomic(target: Path) -> Iterator[Path]:
    tmp = target.with_name(target.name + ".part")
    yield tmp
    tmp.replace(target)


def write_meta(dest: Path, meta: dict[str, Any]) -> None:
    """Write ``meta`` as the ``meta.json`` of the checkpoint in ``dest``."""
    with _atomic(dest / META_FILE) as tmp:
        tmp.write_text(json.dumps(meta, indent=2))


def read_meta(dest: Path) -> dict[str, Any] | None:
    """The ``meta.json`` of the checkpoint in ``dest``, or None if absent or corrupt."""
    try:
        return json.loads((dest / META_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def stage_checkpoint(
    dest: Path,
    step: int,
    *,
    path: str | None = None,
    state_dict: Any = None,
    optimizer_state_dict: Any = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """
    Prepare a checkpoint's files without committing a local copy.

    State dicts are serialized into ``dest``; a ``path`` is used where it is.
    Any ``meta.json`` already in ``dest`` is removed first, so the step does
    not read as a complete local copy until :func:`commit_checkpoint`.

    Parameters
    ----------
    dest : Path
        Checkpoint directory; created if missing.
    step : int
        The training step for this checkpoint.
    path : str or None
        Path to an existing file, saved as ``checkpoint.pt``.
    state_dict : Any or None
        Model state dict, serialized with ``torch.save`` as ``model.pt``.
    optimizer_state_dict : Any or None
        Optimizer state dict, serialized as ``optimizer.pt`` when using
        ``state_dict``.

    Returns
    -------
    entry : dict[str, Any]
        The ``checkpoints.json`` index entry (``step``, ``timestamp``,
        ``files``).
    sources : dict[str, Path]
        Where each of the checkpoint's files is, keyed by file name.

    Raises
    ------
    ValueError
        If neither or both of ``path`` and ``state_dict`` are provided.
    FileNotFoundError
        If ``path`` is not an existing file.
    """
    if (path is None) == (state_dict is None):
        raise ValueError("Exactly one of `path` or `state_dict` must be provided.")
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / META_FILE).unlink(missing_ok=True)

    sources: dict[str, Path] = {}
    if path is not None:
        sources[LEGACY_FILE] = Path(path)
    else:
        import torch

        states = {MODEL_FILE: state_dict}
        if optimizer_state_dict is not None:
            states[OPTIMIZER_FILE] = optimizer_state_dict
        for name, state in states.items():
            with _atomic(dest / name) as tmp:
                torch.save(state, tmp)
            sources[name] = dest / name

    entry: dict[str, Any] = {
        "step": step,
        "timestamp": timestamp,
        "files": [
            {"name": name, "size_bytes": source.stat().st_size}
            for name, source in sources.items()
        ],
    }
    return entry, sources


def commit_checkpoint(
    dest: Path, entry: dict[str, Any], sources: dict[str, Path]
) -> None:
    """
    Complete the local copy of a staged checkpoint.

    Copies in any staged file that lives outside ``dest``, then writes
    ``meta.json``.

    Parameters
    ----------
    dest : Path
        The checkpoint directory passed to :func:`stage_checkpoint`.
    entry : dict[str, Any]
        The index entry returned by :func:`stage_checkpoint`.
    sources : dict[str, Path]
        The file locations returned by :func:`stage_checkpoint`.
    """
    for name, source in sources.items():
        target = dest / name
        if source != target:
            with _atomic(target) as tmp:
                shutil.copyfile(source, tmp)
    write_meta(dest, entry)


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
