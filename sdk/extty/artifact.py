"""Artifact storage for standalone objects (datasets, models, etc.) on S3."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from extty.s3 import (
    S3Config,
    _default_boto_config,
    _download_with_progress,
    _format_bytes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactMeta:
    """Metadata for a stored artifact.

    Parameters
    ----------
    name : str
        Unique name identifying this artifact.
    description : str
        Human-readable description.
    content_type : str
        Either "file" or "directory".
    created_at : str
        ISO timestamp of initial upload.
    updated_at : str
        ISO timestamp of most recent upload.
    total_size_bytes : int
        Total size of all files in the artifact.
    files : list[dict[str, Any]]
        List of file entries, each with "path" and "size_bytes" keys.
    metadata : dict[str, Any]
        User-defined metadata (arbitrary JSON-serializable dict).
    run_project : str or None
        Project name of the run that produced this artifact, if any.
    run_name : str or None
        Run name that produced this artifact, if any.
    """

    name: str
    description: str
    content_type: str
    created_at: str
    updated_at: str
    total_size_bytes: int
    files: list[dict[str, Any]]
    metadata: dict[str, Any]
    run_project: str | None = None
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_size_bytes": self.total_size_bytes,
            "files": self.files,
            "metadata": self.metadata,
        }
        if self.run_project is not None:
            d["run_project"] = self.run_project
        if self.run_name is not None:
            d["run_name"] = self.run_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArtifactMeta:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            content_type=d.get("content_type", "file"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            total_size_bytes=d.get("total_size_bytes", 0),
            files=d.get("files", []),
            metadata=d.get("metadata", {}),
            run_project=d.get("run_project"),
            run_name=d.get("run_name"),
        )


def _build_client(s3_config: S3Config):
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 operations. Install with: pip install extty[s3]"
        )

    kwargs: dict = {"config": _default_boto_config()}
    if s3_config.region:
        kwargs["region_name"] = s3_config.region
    if s3_config.access_key_id and s3_config.secret_access_key:
        kwargs["aws_access_key_id"] = s3_config.access_key_id
        kwargs["aws_secret_access_key"] = s3_config.secret_access_key
    if s3_config.endpoint_url:
        kwargs["endpoint_url"] = s3_config.endpoint_url

    return boto3.client("s3", **kwargs)


def _resolve_config(s3_config: S3Config | None) -> S3Config:
    if s3_config is not None:
        return s3_config
    loaded = S3Config.load()
    if loaded is None:
        raise ValueError(
            "S3 configuration required. Set EXTTY_S3_BUCKET environment variable "
            "or provide s3_config parameter."
        )
    return loaded


def _artifacts_prefix(s3_config: S3Config) -> str:
    base = s3_config.prefix.rstrip("/")
    return f"{base}/artifacts" if base else "artifacts"


def _walk_directory(path: Path) -> list[tuple[str, int]]:
    """Walk a directory and return (relative_path, size_bytes) for each file."""
    entries = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            rel = str(file_path.relative_to(path))
            entries.append((rel, file_path.stat().st_size))
    return entries


def _update_index(
    client,
    bucket: str,
    prefix: str,
    meta: ArtifactMeta,
    *,
    remove: bool = False,
) -> None:
    """Update the artifacts index.json on S3."""
    index_key = f"{prefix}/index.json"
    entries: list[dict[str, Any]] = []

    try:
        response = client.get_object(Bucket=bucket, Key=index_key)
        entries = json.loads(response["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        pass

    entries = [e for e in entries if e.get("name") != meta.name]
    if not remove:
        entries.append(meta.to_dict())
    entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)

    client.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=json.dumps(entries, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def save_artifact(
    name: str,
    path: str | Path,
    *,
    description: str = "",
    metadata: dict[str, Any] | None = None,
    s3_config: S3Config | None = None,
    run_project: str | None = None,
    run_name: str | None = None,
) -> ArtifactMeta:
    """
    Upload a file or directory to S3 as an artifact.

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
    run_project : str, optional
        Project name of the run that produced this artifact.
    run_name : str, optional
        Run name that produced this artifact.

    Returns
    -------
    ArtifactMeta
        Metadata for the saved artifact.
    """
    config = _resolve_config(s3_config)
    client = _build_client(config)
    prefix = _artifacts_prefix(config)
    path = Path(path)
    metadata = metadata or {}

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    now = datetime.now(timezone.utc).isoformat()

    existing = _get_existing_meta(client, config.bucket, prefix, name)
    created_at = existing.created_at if existing else now

    if path.is_file():
        content_type = "file"
        size = path.stat().st_size
        files = [{"path": path.name, "size_bytes": size}]
        s3_key = f"{prefix}/{name}/data/{path.name}"
        client.upload_file(str(path), config.bucket, s3_key)
    else:
        content_type = "directory"
        entries = _walk_directory(path)
        files = [{"path": rel, "size_bytes": sz} for rel, sz in entries]
        size = sum(sz for _, sz in entries)
        for rel, _ in entries:
            s3_key = f"{prefix}/{name}/data/{rel}"
            client.upload_file(str(path / rel), config.bucket, s3_key)

    meta = ArtifactMeta(
        name=name,
        description=description,
        content_type=content_type,
        created_at=created_at,
        updated_at=now,
        total_size_bytes=size,
        files=files,
        metadata=metadata,
        run_project=run_project,
        run_name=run_name,
    )

    client.put_object(
        Bucket=config.bucket,
        Key=f"{prefix}/{name}/meta.json",
        Body=json.dumps(meta.to_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    _update_index(client, config.bucket, prefix, meta)

    logger.info("Saved artifact '%s' (%d bytes)", name, size)
    return meta


def _get_existing_meta(
    client, bucket: str, prefix: str, name: str
) -> ArtifactMeta | None:
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/{name}/meta.json")
        data = json.loads(response["Body"].read().decode("utf-8"))
        return ArtifactMeta.from_dict(data)
    except client.exceptions.NoSuchKey:
        return None


def list_artifacts(
    *,
    s3_config: S3Config | None = None,
) -> list[ArtifactMeta]:
    """
    List all artifacts stored in S3.

    Parameters
    ----------
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.

    Returns
    -------
    list[ArtifactMeta]
        All artifacts, sorted by updated_at descending.
    """
    config = _resolve_config(s3_config)
    client = _build_client(config)
    prefix = _artifacts_prefix(config)
    index_key = f"{prefix}/index.json"

    try:
        response = client.get_object(Bucket=config.bucket, Key=index_key)
        entries = json.loads(response["Body"].read().decode("utf-8"))
        return [ArtifactMeta.from_dict(e) for e in entries]
    except client.exceptions.NoSuchKey:
        pass

    return _list_artifacts_by_prefix(client, config.bucket, prefix)


def _list_artifacts_by_prefix(client, bucket: str, prefix: str) -> list[ArtifactMeta]:
    """Fallback: list artifacts by scanning S3 prefixes."""
    artifacts = []
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/")
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            meta = _get_existing_meta(client, bucket, prefix, name)
            if meta:
                artifacts.append(meta)

    artifacts.sort(key=lambda a: a.updated_at, reverse=True)
    return artifacts


def get_artifact(
    name: str,
    *,
    s3_config: S3Config | None = None,
) -> ArtifactMeta:
    """
    Get metadata for a single artifact.

    Parameters
    ----------
    name : str
        Name of the artifact.
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.

    Returns
    -------
    ArtifactMeta
        Metadata for the artifact.

    Raises
    ------
    KeyError
        If the artifact does not exist.
    """
    config = _resolve_config(s3_config)
    client = _build_client(config)
    prefix = _artifacts_prefix(config)

    meta = _get_existing_meta(client, config.bucket, prefix, name)
    if meta is None:
        raise KeyError(f"Artifact '{name}' not found")
    return meta


def _local_cache_valid(local_dir: Path, files: list[dict[str, Any]]) -> bool:
    for f in files:
        p = local_dir / f["path"]
        if not p.exists() or p.stat().st_size != f["size_bytes"]:
            return False
    return True


def load_artifact(
    name: str,
    dest: str | Path | None = None,
    *,
    cache: bool = False,
    s3_config: S3Config | None = None,
) -> bytes | Path:
    """
    Load an artifact from S3.

    For single-file artifacts with no ``dest``, returns the file contents as
    bytes. Otherwise downloads to disk and returns the path.

    Parameters
    ----------
    name : str
        Name of the artifact.
    dest : str or Path, optional
        Directory to download into. If omitted, single-file artifacts are
        returned as bytes; directory artifacts are cached locally.
    cache : bool, optional
        If True, download into the default local artifacts directory and skip
        the download on subsequent calls when local files match the artifact
        metadata by size. Single-file artifacts are still returned as bytes
        (read from the local copy); directory artifacts return the cache path.
        Mutually exclusive with ``dest``.
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.

    Returns
    -------
    bytes or Path
        File contents as bytes (single file, no dest), or path to
        downloaded file/directory.
    """
    if cache and dest is not None:
        raise ValueError("cache=True is only valid when dest is None")

    config = _resolve_config(s3_config)
    client = _build_client(config)
    prefix = _artifacts_prefix(config)

    meta = _get_existing_meta(client, config.bucket, prefix, name)
    if meta is None:
        raise KeyError(f"Artifact '{name}' not found")

    if cache:
        from extty.storage import get_artifacts_dir

        local_dir = get_artifacts_dir() / name / "data"
        if _local_cache_valid(local_dir, meta.files):
            logger.info(
                "artifact '%s': using cached copy at %s (%s)",
                name,
                local_dir,
                _format_bytes(meta.total_size_bytes),
            )
        else:
            local_dir.mkdir(parents=True, exist_ok=True)
            for file_entry in meta.files:
                s3_key = f"{prefix}/{name}/data/{file_entry['path']}"
                local_path = local_dir / file_entry["path"]
                local_path.parent.mkdir(parents=True, exist_ok=True)
                _download_with_progress(
                    client,
                    config.bucket,
                    s3_key,
                    local_path,
                    label=f"artifact {name} / {file_entry['path']}",
                )

        if meta.content_type == "file":
            return (local_dir / meta.files[0]["path"]).read_bytes()
        return local_dir

    if meta.content_type == "file" and dest is None:
        file_entry = meta.files[0]
        s3_key = f"{prefix}/{name}/data/{file_entry['path']}"
        response = client.get_object(Bucket=config.bucket, Key=s3_key)
        return response["Body"].read()

    if dest is None:
        from extty.storage import get_artifacts_dir

        dest = get_artifacts_dir() / name / "data"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for file_entry in meta.files:
        s3_key = f"{prefix}/{name}/data/{file_entry['path']}"
        local_path = dest / file_entry["path"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _download_with_progress(
            client,
            config.bucket,
            s3_key,
            local_path,
            label=f"artifact {name} / {file_entry['path']}",
        )

    if meta.content_type == "file":
        return dest / meta.files[0]["path"]
    return dest


def delete_artifact(
    name: str,
    *,
    s3_config: S3Config | None = None,
) -> None:
    """
    Delete an artifact from S3.

    Parameters
    ----------
    name : str
        Name of the artifact to delete.
    s3_config : S3Config, optional
        S3 configuration. Loaded from environment if not provided.
    """
    config = _resolve_config(s3_config)
    client = _build_client(config)
    prefix = _artifacts_prefix(config)

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=config.bucket, Prefix=f"{prefix}/{name}/")
    for page in pages:
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=config.bucket, Delete={"Objects": objects})

    meta = ArtifactMeta(
        name=name,
        description="",
        content_type="file",
        created_at="",
        updated_at="",
        total_size_bytes=0,
        files=[],
        metadata={},
    )
    _update_index(client, config.bucket, prefix, meta, remove=True)

    logger.info("Deleted artifact '%s'", name)
