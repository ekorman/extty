"""
Checkpoint layout and lifecycle, shared by the local and S3 backends.

Every save of a checkpoint gets a ``save_id``, recorded in its ``meta.json``.

Locally, a save is written to a private staging directory under
``<run dir>/checkpoints/.staging/`` and moved into
``<run dir>/checkpoints/<step>/`` with one rename, ``meta.json`` included. A
step directory therefore only ever holds a single committed save, or the part
of one that was downloaded, and never a mix of two.

In S3, a save's files go under ``checkpoints/<step>/<save_id>/`` and the
step's ``meta.json`` is written last. It always names a save whose files are
all uploaded, so it is the commit record; ``checkpoints.json`` is only an index
for listing. Checkpoints saved before save IDs existed keep their files
directly under ``checkpoints/<step>/``.

Whether a local copy is safe to delete is decided by :func:`checkpoint_status`,
which compares the local and remote save IDs. The TUI implements the same
table, and both are tested against ``spec/checkpoint_status.json``.
"""

from __future__ import annotations

import enum
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from extty.storage import get_run_dir

META_FILE = "meta.json"
LEGACY_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"
STAGING_DIR = ".staging"
ABANDONED_AFTER_SECONDS = 3600


class CheckpointStatus(enum.Enum):
    """Where a checkpoint's save lives, comparing the local copy with S3."""

    LOCAL_ONLY = "local_only"
    SYNCED = "synced"
    DIVERGED = "diverged"
    REMOTE_ONLY = "remote_only"
    CACHED = "cached"
    UNTRACKED = "untracked"


DELETABLE_LOCALLY = frozenset({CheckpointStatus.SYNCED, CheckpointStatus.CACHED})


@dataclass(frozen=True)
class LocalCopy:
    """
    A checkpoint directory present in the runs dir.

    Attributes
    ----------
    meta : dict[str, Any] or None
        The directory's ``meta.json``. None for a directory without one, which
        only a download made before save IDs existed can leave behind.
    """

    meta: dict[str, Any] | None


@dataclass(frozen=True)
class StagedCheckpoint:
    """
    A save prepared by :func:`stage_checkpoint`, not yet committed anywhere.

    Attributes
    ----------
    meta : dict[str, Any]
        The save's ``meta.json`` entry (``step``, ``save_id``, ``timestamp``,
        ``files``).
    dir : Path
        The private staging directory holding the serialized files.
    sources : dict[str, Path]
        Where each of the save's files is, keyed by file name. A file saved
        from a ``path`` stays where it is until it has to be kept locally.
    """

    meta: dict[str, Any]
    dir: Path
    sources: dict[str, Path]


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    """Directory holding a run's checkpoint for ``step``."""
    return run_dir / "checkpoints" / str(step)


def local_checkpoint_dir(project: str, run_name: str, step: int) -> Path:
    """Directory holding the checkpoint for ``step`` of a run in the runs dir."""
    return checkpoint_dir(get_run_dir(project, run_name), step)


def _owned_name(*suffix: str) -> str:
    """A unique name recording which host and process created it."""
    return "#".join((socket.gethostname(), str(os.getpid()), uuid.uuid4().hex, *suffix))


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_abandoned_staging(staging_root: Path) -> None:
    """
    Delete staging directories left behind by processes that died.

    Only directories created on this host by a process that is no longer
    running, and untouched for :data:`ABANDONED_AFTER_SECONDS`, are removed:
    another process may be using any other one. A directory set aside while
    replacing a step is kept, since it can hold the only copy of a save if the
    replacement failed.
    """
    if os.name != "posix":
        return
    try:
        entries = list(staging_root.iterdir())
    except FileNotFoundError:
        return
    host = socket.gethostname()
    cutoff = time.time() - ABANDONED_AFTER_SECONDS
    for entry in entries:
        parts = entry.name.split("#")
        if len(parts) != 3 or parts[0] != host or not parts[1].isdigit():
            continue
        try:
            abandoned = entry.stat().st_mtime < cutoff
        except FileNotFoundError:
            continue
        if abandoned and not _process_is_running(int(parts[1])):
            shutil.rmtree(entry, ignore_errors=True)


def new_staging_dir(run_dir: Path) -> Path:
    """
    Create an empty, uniquely named staging directory for ``run_dir``.

    Staging directories abandoned by crashed processes are removed first.
    """
    staging_root = run_dir / "checkpoints" / STAGING_DIR
    remove_abandoned_staging(staging_root)
    staging = staging_root / _owned_name()
    staging.mkdir(parents=True)
    return staging


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


def save_identity(meta: dict[str, Any]) -> str:
    """
    The identity of the save a meta entry describes.

    Parameters
    ----------
    meta : dict[str, Any]
        A ``meta.json`` / ``checkpoints.json`` entry.

    Returns
    -------
    str
        The entry's ``save_id``. Entries written before save IDs existed are
        identified by their timestamp: those saves can no longer change.
    """
    save_id = meta.get("save_id")
    if save_id:
        return str(save_id)
    return f"legacy:{meta.get('timestamp')}"


def remote_relpath(meta: dict[str, Any], name: str) -> str:
    """Path of one of a save's files, relative to its step's S3 prefix."""
    save_id = meta.get("save_id")
    return f"{save_id}/{name}" if save_id else name


def read_meta(dest: Path) -> dict[str, Any] | None:
    """The ``meta.json`` of the checkpoint in ``dest``, or None if absent or corrupt."""
    try:
        return json.loads((dest / META_FILE).read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def write_meta(dest: Path, meta: dict[str, Any]) -> None:
    """Write ``meta`` as the ``meta.json`` of the directory ``dest``."""
    (dest / META_FILE).write_text(json.dumps(meta, indent=2))


def read_local_copy(dest: Path) -> LocalCopy | None:
    """The checkpoint directory ``dest``, or None if there is none."""
    if not dest.is_dir():
        return None
    return LocalCopy(meta=read_meta(dest))


def checkpoint_status(
    local: LocalCopy | None, remote: dict[str, Any] | None
) -> CheckpointStatus | None:
    """
    Compare a step's local copy with the save S3 has for it.

    Parameters
    ----------
    local : LocalCopy or None
        The step's directory in the runs dir, if any.
    remote : dict[str, Any] or None
        S3's ``meta.json`` for the step, if S3 has one.

    Returns
    -------
    CheckpointStatus or None
        None when neither side has the step.
    """
    if local is None:
        return None if remote is None else CheckpointStatus.REMOTE_ONLY
    if local.meta is None:
        return CheckpointStatus.UNTRACKED if remote is None else CheckpointStatus.CACHED
    if remote is None:
        return CheckpointStatus.LOCAL_ONLY
    if save_identity(local.meta) == save_identity(remote):
        return CheckpointStatus.SYNCED
    return CheckpointStatus.DIVERGED


def stage_checkpoint(
    run_dir: Path,
    step: int,
    *,
    path: str | None = None,
    state_dict: Any = None,
    optimizer_state_dict: Any = None,
) -> StagedCheckpoint:
    """
    Prepare a new save of a checkpoint without committing it anywhere.

    State dicts are serialized into a new staging directory; a ``path`` is
    used where it is. Nothing in ``<run dir>/checkpoints/<step>/`` changes.

    Parameters
    ----------
    run_dir : Path
        The run's directory.
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
    StagedCheckpoint
        The staged save. Pass it to :func:`discard_staged` once done with it.

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
    staging = new_staging_dir(run_dir)
    sources: dict[str, Path] = {}
    try:
        if path is not None:
            sources[LEGACY_FILE] = Path(path)
        else:
            import torch

            states = {MODEL_FILE: state_dict}
            if optimizer_state_dict is not None:
                states[OPTIMIZER_FILE] = optimizer_state_dict
            for name, state in states.items():
                torch.save(state, staging / name)
                sources[name] = staging / name
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    meta: dict[str, Any] = {
        "step": step,
        "save_id": uuid.uuid4().hex,
        "timestamp": timestamp,
        "files": [
            {"name": name, "size_bytes": source.stat().st_size}
            for name, source in sources.items()
        ],
    }
    return StagedCheckpoint(meta=meta, dir=staging, sources=sources)


def _set_aside(dest: Path) -> Path:
    """Rename ``dest`` into the staging area, returning its new path."""
    aside = dest.parent / STAGING_DIR / _owned_name("replaced")
    aside.parent.mkdir(parents=True, exist_ok=True)
    dest.rename(aside)
    return aside


def publish(staging: Path, dest: Path) -> None:
    """
    Move a complete staging directory into place as ``dest``.

    Any directory already at ``dest`` is replaced. ``dest`` never holds a mix
    of the two: the old directory is renamed away before the new one is
    renamed in, and renamed back if that fails.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(dest)
        return
    aside = _set_aside(dest)
    try:
        staging.rename(dest)
    except BaseException:
        aside.rename(dest)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _publish_if_vacant(staging: Path, dest: Path) -> bool:
    """
    Move ``staging`` into place as ``dest`` unless ``dest`` holds a save.

    A step directory without ``meta.json`` is an old download cache and is
    replaced. If another process publishes to ``dest`` first, it wins and
    ``staging`` is left where it is.

    Returns
    -------
    bool
        True if ``staging`` is now at ``dest``.
    """
    if dest.exists():
        if (dest / META_FILE).exists():
            return False
        try:
            shutil.rmtree(_set_aside(dest), ignore_errors=True)
        except FileNotFoundError:
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(dest)
    except OSError:
        if dest.exists():
            return False
        raise
    return True


def adopt_download(staging: Path, dest: Path, meta: dict[str, Any]) -> bool:
    """
    Make downloaded files of a save part of the local copy of its step.

    Safe to call from several processes loading the same step at once: the
    first to publish wins, and the others add any files it lacks.

    Parameters
    ----------
    staging : Path
        A staging directory holding ``meta.json`` and some of the save's files.
    dest : Path
        The step's directory in the runs dir.
    meta : dict[str, Any]
        The save's meta entry.

    Returns
    -------
    bool
        True if ``dest`` now holds the save with the staged files. False if it
        holds a different save, which is left alone.
    """
    if _publish_if_vacant(staging, dest):
        return True
    current = read_meta(dest)
    if current is None or save_identity(current) != save_identity(meta):
        return False
    for staged_file in staging.iterdir():
        if staged_file.name != META_FILE and not (dest / staged_file.name).exists():
            os.replace(staged_file, dest / staged_file.name)
    return True


def commit_checkpoint(run_dir: Path, staged: StagedCheckpoint) -> None:
    """
    Make a staged save the run's local copy of its step.

    Copies in a file saved from a ``path``, writes ``meta.json``, and replaces
    ``<run dir>/checkpoints/<step>/`` with the staging directory.
    """
    for name, source in staged.sources.items():
        if source.parent != staged.dir:
            shutil.copyfile(source, staged.dir / name)
    write_meta(staged.dir, staged.meta)
    publish(staged.dir, checkpoint_dir(run_dir, staged.meta["step"]))


def discard_staged(staged: StagedCheckpoint) -> None:
    """Remove a staged save's staging directory, if it is still there."""
    shutil.rmtree(staged.dir, ignore_errors=True)


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
    Load the checkpoint in ``dest`` if it has every file this load needs.

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
