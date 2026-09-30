//! Checkpoint identity and local-vs-S3 status, mirroring `extty.checkpoints`.
//!
//! Every save of a checkpoint has a `save_id` in its `meta.json`. Locally, a
//! step directory holds one committed save, or the downloaded part of one, and
//! is only ever replaced whole, by renaming a staging directory into place. In
//! S3, a save's files live under `checkpoints/<step>/<save_id>/` and the step's
//! `meta.json`, written last, names the committed save. The SDK and this module
//! are both tested against `spec/checkpoint_status.json`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::Value;

pub const META_FILE: &str = "meta.json";
pub const STAGING_DIR: &str = ".staging";
const LEGACY_FILE: &str = "checkpoint.pt";

/// A checkpoint directory present in the runs dir.
#[derive(Debug, Clone)]
pub enum LocalCopy {
    /// A committed save or a download, described by its `meta.json`.
    Committed(Value),
    /// Files without a `meta.json`, left by downloads made before save IDs.
    Untracked,
}

/// How a step's local copy compares with the save S3 has for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    LocalOnly,
    Synced,
    Diverged,
    RemoteOnly,
    Cached,
    Untracked,
}

impl Status {
    /// Whether the local copy can be deleted without losing a save.
    pub fn local_is_deletable(self) -> bool {
        matches!(self, Status::Synced | Status::Cached)
    }

    /// The name used for this status in `spec/checkpoint_status.json`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn as_str(self) -> &'static str {
        match self {
            Status::LocalOnly => "local_only",
            Status::Synced => "synced",
            Status::Diverged => "diverged",
            Status::RemoteOnly => "remote_only",
            Status::Cached => "cached",
            Status::Untracked => "untracked",
        }
    }
}

/// The identity of the save a meta entry describes.
///
/// Entries written before save IDs existed are identified by their timestamp:
/// those saves can no longer change.
pub fn save_identity(meta: &Value) -> String {
    match meta.get("save_id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => format!("legacy:{}", meta.get("timestamp").unwrap_or(&Value::Null)),
    }
}

/// Path of one of a save's files, relative to its step's S3 prefix.
pub fn remote_relpath(meta: &Value, name: &str) -> String {
    match meta.get("save_id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() => format!("{}/{}", id, name),
        _ => name.to_string(),
    }
}

/// Names of the files recorded in a meta entry, or `checkpoint.pt` if none are.
pub fn file_names(meta: &Value) -> Vec<String> {
    let names: Vec<String> = meta
        .get("files")
        .and_then(Value::as_array)
        .map(|files| {
            files
                .iter()
                .filter_map(|f| f.as_str().or_else(|| f.get("name")?.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if names.is_empty() {
        vec![LEGACY_FILE.to_string()]
    } else {
        names
    }
}

/// The checkpoint directory `step_dir`, or `None` if there is none.
pub fn read_local_copy(step_dir: &Path) -> Option<LocalCopy> {
    if !step_dir.is_dir() {
        return None;
    }
    let meta = fs::read(step_dir.join(META_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    Some(match meta {
        Some(meta) => LocalCopy::Committed(meta),
        None => LocalCopy::Untracked,
    })
}

/// Compare a step's local copy with S3's `meta.json` for it.
pub fn status(local: Option<&LocalCopy>, remote: Option<&Value>) -> Option<Status> {
    match (local, remote) {
        (None, None) => None,
        (None, Some(_)) => Some(Status::RemoteOnly),
        (Some(LocalCopy::Untracked), None) => Some(Status::Untracked),
        (Some(LocalCopy::Untracked), Some(_)) => Some(Status::Cached),
        (Some(LocalCopy::Committed(_)), None) => Some(Status::LocalOnly),
        (Some(LocalCopy::Committed(local)), Some(remote)) => {
            Some(if save_identity(local) == save_identity(remote) {
                Status::Synced
            } else {
                Status::Diverged
            })
        }
    }
}

/// The run's local checkpoint directories, keyed by step.
pub fn local_step_dirs(run_dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(entries) = fs::read_dir(run_dir.join("checkpoints")) else {
        return Vec::new();
    };
    let mut dirs: Vec<(u64, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| {
            let step = p.file_name()?.to_str()?.parse::<u64>().ok()?;
            Some((step, p))
        })
        .collect();
    dirs.sort();
    dirs
}

fn unique_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("tui-{}-{}", std::process::id(), nanos)
}

/// Create an empty, uniquely named staging directory for `run_dir`.
pub fn new_staging_dir(run_dir: &Path) -> Result<PathBuf> {
    let staging = run_dir
        .join("checkpoints")
        .join(STAGING_DIR)
        .join(unique_name());
    fs::create_dir_all(&staging)
        .with_context(|| format!("Failed to create {}", staging.display()))?;
    Ok(staging)
}

/// Move a complete staging directory into place as `dest`, replacing any
/// directory there. `dest` never holds a mix of the two: the old directory is
/// renamed away before the new one is renamed in, and renamed back if that
/// fails.
pub fn publish(staging: &Path, dest: &Path) -> Result<()> {
    let parent = dest
        .parent()
        .context("checkpoint directory has no parent")?;
    if !dest.exists() {
        fs::create_dir_all(parent)?;
        fs::rename(staging, dest)?;
        return Ok(());
    }
    let aside_dir = parent.join(STAGING_DIR);
    fs::create_dir_all(&aside_dir)?;
    let aside = aside_dir.join(format!("{}.replaced", unique_name()));
    fs::rename(dest, &aside)?;
    if let Err(e) = fs::rename(staging, dest) {
        fs::rename(&aside, dest).with_context(|| {
            format!(
                "Failed to restore {} from {}",
                dest.display(),
                aside.display()
            )
        })?;
        return Err(e.into());
    }
    let _ = fs::remove_dir_all(&aside);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = include_str!("../../spec/checkpoint_status.json");

    #[test]
    fn status_matches_spec() {
        let spec: Value = serde_json::from_str(SPEC).unwrap();
        for case in spec["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let local = match &case["local"] {
                Value::Null => None,
                Value::String(s) if s == "untracked" => Some(LocalCopy::Untracked),
                meta => Some(LocalCopy::Committed(meta.clone())),
            };
            let remote = Some(&case["remote"]).filter(|r| !r.is_null());

            let got = status(local.as_ref(), remote);

            assert_eq!(
                got.map(Status::as_str),
                case["status"].as_str(),
                "status for {name}"
            );
            assert_eq!(
                got.is_some_and(Status::local_is_deletable),
                case["deletable"].as_bool().unwrap(),
                "deletable for {name}"
            );
        }
    }

    #[test]
    fn remote_paths_use_the_save_prefix() {
        let current: Value = serde_json::json!({"save_id": "abc"});
        let legacy: Value = serde_json::json!({"timestamp": "t"});
        assert_eq!(remote_relpath(&current, "model.pt"), "abc/model.pt");
        assert_eq!(remote_relpath(&legacy, "model.pt"), "model.pt");
    }

    #[test]
    fn file_names_handle_every_entry_shape() {
        let dicts = serde_json::json!({"files": [{"name": "model.pt"}, {"name": "optimizer.pt"}]});
        let bare = serde_json::json!({"files": ["checkpoint.pt"]});
        let none = serde_json::json!({"step": 1});
        assert_eq!(file_names(&dicts), ["model.pt", "optimizer.pt"]);
        assert_eq!(file_names(&bare), ["checkpoint.pt"]);
        assert_eq!(file_names(&none), ["checkpoint.pt"]);
    }

    #[test]
    fn publish_replaces_the_whole_directory() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let dest = run_dir.join("checkpoints/5");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("optimizer.pt"), b"old").unwrap();
        let staging = new_staging_dir(run_dir).unwrap();
        fs::write(staging.join("model.pt"), b"new").unwrap();

        publish(&staging, &dest).unwrap();

        let names: Vec<String> = fs::read_dir(&dest)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, ["model.pt"]);
        assert!(!staging.exists());
        assert_eq!(
            fs::read_dir(run_dir.join("checkpoints").join(STAGING_DIR))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn failed_publish_restores_the_old_directory() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("checkpoints/5");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("model.pt"), b"old").unwrap();
        let missing_staging = dir.path().join("checkpoints/.staging/gone");

        assert!(publish(&missing_staging, &dest).is_err());

        assert_eq!(fs::read(dest.join("model.pt")).unwrap(), b"old");
    }

    #[test]
    fn local_step_dirs_skip_staging() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("checkpoints/20")).unwrap();
        fs::create_dir_all(dir.path().join("checkpoints/3")).unwrap();
        new_staging_dir(dir.path()).unwrap();

        let steps: Vec<u64> = local_step_dirs(dir.path())
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        assert_eq!(steps, [3, 20]);
    }
}
