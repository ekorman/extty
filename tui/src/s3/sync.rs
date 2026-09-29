use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Credentials;
use aws_sdk_s3::types::Object;
use futures_util::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use super::config::S3Config;

/// Maximum number of objects fetched concurrently by [`S3Client::download_run`].
const DOWNLOAD_CONCURRENCY: usize = 16;

/// Outcome of a [`S3Client::download_run`] that was not short-circuited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PullSummary {
    /// Objects fetched from S3.
    pub downloaded: usize,
    /// Objects whose local copy was already current and left untouched.
    pub skipped: usize,
}

#[derive(Debug, Clone)]
pub struct RemoteRun {
    pub project: String,
    pub name: String,
}

pub struct S3Client {
    client: Client,
    config: S3Config,
}

impl S3Client {
    pub async fn new(config: S3Config) -> Result<Self> {
        let mut aws_config_builder = aws_config::defaults(aws_config::BehaviorVersion::latest());

        let region = config
            .region
            .clone()
            .unwrap_or_else(|| "us-east-1".to_string());
        aws_config_builder = aws_config_builder.region(aws_config::Region::new(region));

        if let (Some(access_key), Some(secret_key)) =
            (&config.access_key_id, &config.secret_access_key)
        {
            aws_config_builder = aws_config_builder.credentials_provider(Credentials::new(
                access_key.clone(),
                secret_key.clone(),
                None,
                None,
                "extty",
            ));
        }

        let aws_config = aws_config_builder.load().await;
        let mut s3_config_builder =
            aws_sdk_s3::config::Builder::from(&aws_config).force_path_style(true);

        if let Some(endpoint) = &config.endpoint_url {
            s3_config_builder = s3_config_builder.endpoint_url(endpoint);
        }

        let client = Client::from_conf(s3_config_builder.build());

        Ok(Self { client, config })
    }

    fn s3_prefix(&self, path: &str) -> String {
        if self.config.prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", self.config.prefix, path)
        }
    }

    pub async fn list_runs(&self, project: Option<&str>) -> Result<Vec<RemoteRun>> {
        let projects = match project {
            Some(p) => vec![p.to_string()],
            None => self.list_projects().await?,
        };

        let mut runs = Vec::new();
        for proj in projects {
            let prefix = self.s3_prefix(&format!("runs/{}/", proj));
            let mut continuation_token: Option<String> = None;

            loop {
                let mut request = self
                    .client
                    .list_objects_v2()
                    .bucket(&self.config.bucket)
                    .prefix(&prefix)
                    .delimiter("/");

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                let response = request.send().await.context("Failed to list S3 objects")?;

                for cp in response.common_prefixes() {
                    if let Some(prefix_str) = cp.prefix() {
                        let name = prefix_str
                            .trim_end_matches('/')
                            .rsplit('/')
                            .next()
                            .unwrap_or("")
                            .to_string();
                        if !name.is_empty() {
                            runs.push(RemoteRun {
                                project: proj.clone(),
                                name,
                            });
                        }
                    }
                }

                if response.is_truncated() == Some(true) {
                    continuation_token = response.next_continuation_token().map(|s| s.to_string());
                } else {
                    break;
                }
            }
        }

        Ok(runs)
    }

    async fn list_projects(&self) -> Result<Vec<String>> {
        let prefix = self.s3_prefix("runs/");
        let mut projects = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix)
                .delimiter("/");

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await.context("Failed to list S3 projects")?;

            for cp in response.common_prefixes() {
                if let Some(prefix_str) = cp.prefix() {
                    let name = prefix_str
                        .trim_end_matches('/')
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    if !name.is_empty() {
                        projects.push(name);
                    }
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(projects)
    }

    /// Downloads a run from S3.
    ///
    /// Returns `Ok(None)` if the run was skipped because it is already completed
    /// locally. Otherwise every object under the run prefix (checkpoints aside) is
    /// compared against its local copy and only stale or missing ones are fetched,
    /// up to [`DOWNLOAD_CONCURRENCY`] at a time. Pass `force` to re-download
    /// everything.
    pub async fn download_run(
        &self,
        project: &str,
        run: &str,
        dest: &Path,
        force: bool,
        dry_run: bool,
    ) -> Result<Option<PullSummary>> {
        let prefix = self.s3_prefix(&format!("runs/{}/{}/", project, run));
        let run_dir = dest.join(project).join(run);

        if !force && !dry_run {
            let meta_path = run_dir.join("meta.json");
            if let Ok(content) = fs::read(&meta_path)
                && let Ok(meta) = serde_json::from_slice::<MetaJson>(&content)
                && meta.status.as_deref() == Some("completed")
            {
                let _ = self
                    .sync_checkpoints_index_if_newer(project, run, &run_dir)
                    .await;
                return Ok(None);
            }
        }

        if dry_run {
            println!(
                "Would download: {}/{} -> {}",
                project,
                run,
                run_dir.display()
            );
            return Ok(Some(PullSummary::default()));
        }

        fs::create_dir_all(&run_dir)?;

        let mut pending: Vec<(String, PathBuf)> = Vec::new();
        let mut skipped = 0usize;
        for object in self.list_objects(&prefix).await? {
            let Some(key) = object.key() else {
                continue;
            };
            let relative_path = key.strip_prefix(&prefix).unwrap_or(key);
            if relative_path.is_empty() || relative_path.starts_with("checkpoints/") {
                continue;
            }

            let local_path = run_dir.join(relative_path);
            if !force && local_is_current(&local_path, &object, is_mergeable(key)) {
                skipped += 1;
                continue;
            }
            if let Some(parent) = local_path.parent() {
                fs::create_dir_all(parent)?;
            }
            pending.push((key.to_string(), local_path));
        }

        let downloaded = pending.len();
        futures_util::stream::iter(pending)
            .map(move |(key, local_path)| async move {
                self.download_file(&key, &local_path, force).await
            })
            .buffer_unordered(DOWNLOAD_CONCURRENCY)
            .try_collect::<()>()
            .await?;

        Ok(Some(PullSummary {
            downloaded,
            skipped,
        }))
    }

    async fn list_objects(&self, prefix: &str) -> Result<Vec<Object>> {
        let mut objects = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;
            objects.extend(response.contents().iter().cloned());

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(objects)
    }

    async fn download_file(&self, key: &str, local_path: &Path, force: bool) -> Result<()> {
        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await?;

        let body = response.body.collect().await?.into_bytes();

        if !force && local_path.exists() {
            if key.ends_with(".csv") {
                merge_csv_file(local_path, &body)?;
            } else if key.ends_with(".jsonl") {
                merge_jsonl_file(local_path, &body)?;
            } else if key.ends_with("meta.json") {
                merge_meta_file(local_path, &body)?;
            } else {
                fs::write(local_path, &body)?;
            }
        } else {
            fs::write(local_path, &body)?;
        }

        Ok(())
    }

    pub async fn upload_run(
        &self,
        project: &str,
        run: &str,
        source: &Path,
        force: bool,
        dry_run: bool,
    ) -> Result<()> {
        let run_dir = source.join(project).join(run);
        if !run_dir.exists() {
            anyhow::bail!("Run directory does not exist: {}", run_dir.display());
        }

        if dry_run {
            println!("Would upload: {} -> {}/{}", run_dir.display(), project, run);
            return Ok(());
        }

        let prefix = self.s3_prefix(&format!("runs/{}/{}", project, run));

        for entry in run_files_to_upload(&run_dir)? {
            let relative_path = entry.strip_prefix(&run_dir)?;
            let s3_key = format!("{}/{}", prefix, relative_path.display());

            self.upload_file(&entry, &s3_key, force).await?;
        }

        Ok(())
    }

    async fn upload_file(&self, local_path: &Path, key: &str, force: bool) -> Result<()> {
        let local_content = fs::read(local_path)?;

        let content_to_upload = if !force {
            let existing = self.get_object_content(key).await;
            if let Ok(Some(existing_content)) = existing {
                if key.ends_with(".csv") {
                    merge_csv_bytes(&local_content, &existing_content)?
                } else if key.ends_with(".jsonl") {
                    merge_jsonl_bytes(&local_content, &existing_content)?
                } else if key.ends_with("meta.json") {
                    merge_meta_bytes(&local_content, &existing_content)?
                } else {
                    local_content
                }
            } else {
                local_content
            }
        } else {
            local_content
        };

        let content_type = if key.ends_with(".csv") {
            "text/csv"
        } else if key.ends_with(".json") {
            "application/json"
        } else if key.ends_with(".jsonl") {
            "application/x-ndjson"
        } else {
            "application/octet-stream"
        };

        self.client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .body(content_to_upload.into())
            .content_type(content_type)
            .send()
            .await?;

        Ok(())
    }

    async fn get_object_content(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(response) => {
                let body = response.body.collect().await?.into_bytes();
                Ok(Some(body.to_vec()))
            }
            Err(e) => {
                if e.to_string().contains("NoSuchKey") {
                    Ok(None)
                } else {
                    Err(e.into())
                }
            }
        }
    }

    pub async fn download_checkpoint(
        &self,
        project: &str,
        run: &str,
        step: u64,
        dest: &Path,
        files: Option<&[&str]>,
    ) -> Result<()> {
        let run_dir = dest.join(project).join(run);
        let run_prefix = self.s3_prefix(&format!("runs/{}/{}/", project, run));

        if let Some(file_list) = files {
            for filename in file_list {
                let key = self.s3_prefix(&format!(
                    "runs/{}/{}/checkpoints/{}/{}",
                    project, run, step, filename
                ));
                let relative_path = key.strip_prefix(&run_prefix).unwrap_or(&key);
                let local_path = run_dir.join(relative_path);
                if let Some(parent) = local_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                self.download_file(&key, &local_path, true).await?;
            }
        } else {
            let prefix = self.s3_prefix(&format!("runs/{}/{}/checkpoints/{}/", project, run, step));
            let mut continuation_token: Option<String> = None;

            loop {
                let mut request = self
                    .client
                    .list_objects_v2()
                    .bucket(&self.config.bucket)
                    .prefix(&prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                let response = request.send().await?;

                for object in response.contents() {
                    if let Some(key) = object.key() {
                        let relative_path = key.strip_prefix(&run_prefix).unwrap_or(key);
                        if relative_path.is_empty() {
                            continue;
                        }

                        let local_path = run_dir.join(relative_path);
                        if let Some(parent) = local_path.parent() {
                            fs::create_dir_all(parent)?;
                        }

                        self.download_file(key, &local_path, true).await?;
                    }
                }

                if response.is_truncated() == Some(true) {
                    continuation_token = response.next_continuation_token().map(|s| s.to_string());
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    pub async fn delete_run(&self, project: &str, run: &str) -> Result<()> {
        let prefix = self.s3_prefix(&format!("runs/{}/{}/", project, run));
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list S3 objects for deletion")?;

            let keys: Vec<String> = response
                .contents()
                .iter()
                .filter_map(|obj| obj.key().map(|k| k.to_string()))
                .collect();

            for chunk in keys.chunks(1000) {
                let objects: Vec<_> = chunk
                    .iter()
                    .map(|key| {
                        aws_sdk_s3::types::ObjectIdentifier::builder()
                            .key(key)
                            .build()
                            .unwrap()
                    })
                    .collect();

                let delete = aws_sdk_s3::types::Delete::builder()
                    .set_objects(Some(objects))
                    .build()?;

                self.client
                    .delete_objects()
                    .bucket(&self.config.bucket)
                    .delete(delete)
                    .send()
                    .await
                    .context("Failed to delete S3 objects")?;
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(())
    }

    pub async fn move_run(&self, old_project: &str, new_project: &str, run: &str) -> Result<()> {
        let old_prefix = self.s3_prefix(&format!("runs/{}/{}/", old_project, run));
        let new_prefix = self.s3_prefix(&format!("runs/{}/{}/", new_project, run));
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&old_prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list S3 objects for move")?;

            for object in response.contents() {
                if let Some(key) = object.key() {
                    let relative = key.strip_prefix(&old_prefix).unwrap_or(key);
                    let new_key = format!("{}{}", new_prefix, relative);

                    self.client
                        .copy_object()
                        .bucket(&self.config.bucket)
                        .copy_source(format!("{}/{}", self.config.bucket, key))
                        .key(&new_key)
                        .send()
                        .await
                        .context("Failed to copy S3 object")?;
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        self.delete_run(old_project, run).await?;
        Ok(())
    }

    pub async fn sync_run(
        &self,
        project: &str,
        run: &str,
        local_dir: &Path,
        dry_run: bool,
    ) -> Result<()> {
        let _ = self
            .download_run(project, run, local_dir, false, dry_run)
            .await?;
        self.upload_run(project, run, local_dir, false, dry_run)
            .await?;
        Ok(())
    }
}

/// Keys whose local copy is merged with the remote rather than overwritten.
/// Must agree with the branches in [`S3Client::download_file`].
fn is_mergeable(key: &str) -> bool {
    key.ends_with(".csv") || key.ends_with(".jsonl") || key.ends_with("meta.json")
}

/// Whether the local copy of `object` can be left alone.
///
/// A file is current when it was written after the remote object was last
/// modified (local mtimes are set at download or merge time, so a remote rewrite
/// always lands strictly later) and, for write-once blobs such as logged images,
/// when it has the same byte length. Mergeable text files are re-serialised on
/// merge and so are legitimately a different size from the remote object.
fn local_is_current(local_path: &Path, object: &Object, mergeable: bool) -> bool {
    let Ok(metadata) = fs::metadata(local_path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    let Some(remote_modified) = object.last_modified() else {
        return false;
    };
    let Some(local_modified) = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
    else {
        return false;
    };
    if remote_modified.secs() >= local_modified.as_secs() as i64 {
        return false;
    }
    mergeable || matches!(object.size(), Some(size) if i64::try_from(metadata.len()) == Ok(size))
}

/// Files of a local run that a run upload should push.
///
/// Checkpoints are excluded: they reach S3 only through the SDK's
/// `save_checkpoint`, which uploads the files, their `meta.json` and the index
/// together. Locally, `checkpoints/` can hold multi-GB payloads that were never
/// meant to be uploaded (and partial `.part` files), and `checkpoints.json` can
/// list steps that exist only on this machine.
fn run_files_to_upload(run_dir: &Path) -> Result<Vec<PathBuf>> {
    Ok(walkdir(run_dir)?
        .into_iter()
        .filter(|path| {
            path.strip_prefix(run_dir).is_ok_and(|relative| {
                !relative.starts_with("checkpoints") && relative != Path::new("checkpoints.json")
            })
        })
        .collect())
}

fn walkdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(walkdir(&path)?);
        } else {
            files.push(path);
        }
    }
    Ok(files)
}

fn merge_csv_file(local_path: &Path, remote_content: &[u8]) -> Result<()> {
    let local_content = fs::read(local_path)?;
    let merged = merge_csv_bytes(&local_content, remote_content)?;
    fs::write(local_path, merged)?;
    Ok(())
}

fn merge_csv_bytes(local: &[u8], remote: &[u8]) -> Result<Vec<u8>> {
    let local_str = std::str::from_utf8(local)?;
    let remote_str = std::str::from_utf8(remote)?;

    let mut existing_keys: HashSet<(String, String)> = HashSet::new();
    let mut rows: Vec<HashMap<String, String>> = Vec::new();
    let mut headers: Option<Vec<String>> = None;

    for (content, _) in [(remote_str, "remote"), (local_str, "local")] {
        let mut reader = csv::Reader::from_reader(content.as_bytes());
        if headers.is_none() {
            headers = Some(reader.headers()?.iter().map(|s| s.to_string()).collect());
        }

        for result in reader.deserialize::<HashMap<String, String>>() {
            let row = result?;
            let key = (
                row.get("step").cloned().unwrap_or_default(),
                row.get("timestamp").cloned().unwrap_or_default(),
            );
            if !existing_keys.contains(&key) {
                existing_keys.insert(key);
                rows.push(row);
            }
        }
    }

    rows.sort_by(|a, b| {
        let step_a: f64 = a.get("step").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let step_b: f64 = b.get("step").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let ts_a: f64 = a
            .get("timestamp")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let ts_b: f64 = b
            .get("timestamp")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        step_a
            .partial_cmp(&step_b)
            .unwrap()
            .then(ts_a.partial_cmp(&ts_b).unwrap())
    });

    let mut output = Vec::new();
    {
        let mut writer = csv::Writer::from_writer(&mut output);
        if let Some(ref hdrs) = headers {
            writer.write_record(hdrs)?;
            for row in rows {
                let record: Vec<String> = hdrs
                    .iter()
                    .map(|h| row.get(h).cloned().unwrap_or_default())
                    .collect();
                writer.write_record(&record)?;
            }
        }
        writer.flush()?;
    }

    Ok(output)
}

fn merge_jsonl_file(local_path: &Path, remote_content: &[u8]) -> Result<()> {
    let local_content = fs::read(local_path)?;
    let merged = merge_jsonl_bytes(&local_content, remote_content)?;
    fs::write(local_path, merged)?;
    Ok(())
}

fn merge_jsonl_bytes(local: &[u8], remote: &[u8]) -> Result<Vec<u8>> {
    let local_str = std::str::from_utf8(local)?;
    let remote_str = std::str::from_utf8(remote)?;

    let mut existing_keys: HashSet<(i64, i64)> = HashSet::new();
    let mut records: Vec<serde_json::Value> = Vec::new();

    for content in [remote_str, local_str] {
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value = serde_json::from_str(line)?;
            let step = record.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
            let timestamp = record
                .get("timestamp")
                .and_then(|v| v.as_f64())
                .map(|f| (f * 1_000_000.0) as i64)
                .unwrap_or(0);
            let key = (step, timestamp);
            if !existing_keys.contains(&key) {
                existing_keys.insert(key);
                records.push(record);
            }
        }
    }

    records.sort_by(|a, b| {
        let step_a = a.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
        let step_b = b.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
        let ts_a = a.get("timestamp").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ts_b = b.get("timestamp").and_then(|v| v.as_f64()).unwrap_or(0.0);
        step_a
            .cmp(&step_b)
            .then(ts_a.partial_cmp(&ts_b).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut output = Vec::new();
    for record in records {
        serde_json::to_writer(&mut output, &record)?;
        output.push(b'\n');
    }

    Ok(output)
}

fn merge_meta_file(local_path: &Path, remote_content: &[u8]) -> Result<()> {
    let local_content = fs::read(local_path)?;
    let merged = merge_meta_bytes(&local_content, remote_content)?;
    fs::write(local_path, merged)?;
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct MetaJson {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    finished_at: Option<String>,
    #[serde(flatten)]
    other: HashMap<String, serde_json::Value>,
}

pub struct RemoteArtifact {
    pub name: String,
}

impl S3Client {
    pub async fn list_artifacts(&self) -> Result<Vec<RemoteArtifact>> {
        let prefix = self.s3_prefix("artifacts/");
        let mut artifacts = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix)
                .delimiter("/");

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await.context("Failed to list artifacts")?;

            for cp in response.common_prefixes() {
                if let Some(prefix_str) = cp.prefix() {
                    let name = prefix_str
                        .trim_end_matches('/')
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    if !name.is_empty() {
                        artifacts.push(RemoteArtifact { name });
                    }
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(artifacts)
    }

    pub async fn download_artifact_meta(&self, name: &str, dest: &Path) -> Result<bool> {
        let key = self.s3_prefix(&format!("artifacts/{}/meta.json", name));

        let response = match self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                if e.as_service_error()
                    .map(|se| se.is_no_such_key())
                    .unwrap_or(false)
                {
                    return Ok(false);
                }
                return Err(e).context("Failed to download artifact meta");
            }
        };

        let body = response.body.collect().await?.into_bytes();
        let artifact_dir = dest.join(name);
        fs::create_dir_all(&artifact_dir)?;
        fs::write(artifact_dir.join("meta.json"), &body)?;

        Ok(true)
    }

    pub async fn download_all_artifact_metas(&self, dest: &Path) -> Result<usize> {
        let artifacts = self.list_artifacts().await?;
        let mut count = 0;
        for artifact in artifacts {
            if let Ok(true) = self.download_artifact_meta(&artifact.name, dest).await {
                count += 1;
            }
        }
        Ok(count)
    }

    pub async fn delete_artifact(&self, name: &str) -> Result<()> {
        let prefix = self.s3_prefix(&format!("artifacts/{}/", name));
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list artifact objects for deletion")?;

            let keys: Vec<String> = response
                .contents()
                .iter()
                .filter_map(|obj| obj.key().map(|k| k.to_string()))
                .collect();

            for chunk in keys.chunks(1000) {
                let objects: Vec<_> = chunk
                    .iter()
                    .map(|key| {
                        aws_sdk_s3::types::ObjectIdentifier::builder()
                            .key(key)
                            .build()
                            .unwrap()
                    })
                    .collect();

                let delete = aws_sdk_s3::types::Delete::builder()
                    .set_objects(Some(objects))
                    .build()?;

                self.client
                    .delete_objects()
                    .bucket(&self.config.bucket)
                    .delete(delete)
                    .send()
                    .await
                    .context("Failed to delete artifact objects")?;
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(())
    }

    async fn sync_checkpoints_index_if_newer(
        &self,
        project: &str,
        run: &str,
        run_dir: &Path,
    ) -> Result<bool> {
        let key = self.s3_prefix(&format!("runs/{}/{}/checkpoints.json", project, run));
        let local_path = run_dir.join("checkpoints.json");

        let head = match self
            .client
            .head_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(h) => h,
            Err(e) => {
                if format!("{:?}", e).contains("NotFound") {
                    return Ok(false);
                }
                return Err(e.into());
            }
        };

        if local_path.exists()
            && let Some(remote_dt) = head.last_modified()
            && let Ok(metadata) = fs::metadata(&local_path)
            && let Ok(local_mtime) = metadata.modified()
            && let Ok(local_dur) = local_mtime.duration_since(std::time::UNIX_EPOCH)
            && remote_dt.secs() <= local_dur.as_secs() as i64
        {
            return Ok(false);
        }

        let resp = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await?;
        let body = resp.body.collect().await?.into_bytes();
        fs::create_dir_all(run_dir)?;
        fs::write(&local_path, &body)?;
        Ok(true)
    }

    pub async fn list_remote_checkpoint_steps(&self, project: &str, run: &str) -> Result<Vec<u64>> {
        let index_key = self.s3_prefix(&format!("runs/{}/{}/checkpoints.json", project, run));
        let Some(content) = self.get_object_content(&index_key).await? else {
            return Ok(Vec::new());
        };
        let entries: Vec<serde_json::Value> =
            serde_json::from_slice(&content).context("Failed to parse remote checkpoints.json")?;
        Ok(entries
            .iter()
            .filter_map(|e| e.get("step").and_then(|v| v.as_u64()))
            .collect())
    }

    pub async fn delete_checkpoint(&self, project: &str, run: &str, step: u64) -> Result<()> {
        let prefix = self.s3_prefix(&format!("runs/{}/{}/checkpoints/{}/", project, run, step));
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list checkpoint objects for deletion")?;

            let keys: Vec<String> = response
                .contents()
                .iter()
                .filter_map(|obj| obj.key().map(|k| k.to_string()))
                .collect();

            for chunk in keys.chunks(1000) {
                let objects: Vec<_> = chunk
                    .iter()
                    .map(|key| {
                        aws_sdk_s3::types::ObjectIdentifier::builder()
                            .key(key)
                            .build()
                            .unwrap()
                    })
                    .collect();

                let delete = aws_sdk_s3::types::Delete::builder()
                    .set_objects(Some(objects))
                    .build()?;

                self.client
                    .delete_objects()
                    .bucket(&self.config.bucket)
                    .delete(delete)
                    .send()
                    .await
                    .context("Failed to delete checkpoint objects")?;
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        let index_key = self.s3_prefix(&format!("runs/{}/{}/checkpoints.json", project, run));
        if let Ok(Some(content)) = self.get_object_content(&index_key).await
            && let Ok(mut entries) = serde_json::from_slice::<Vec<serde_json::Value>>(&content)
        {
            entries.retain(|e| e.get("step").and_then(|v| v.as_u64()) != Some(step));
            let updated =
                serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string());
            let _ = self
                .client
                .put_object()
                .bucket(&self.config.bucket)
                .key(&index_key)
                .body(updated.into_bytes().into())
                .content_type("application/json")
                .send()
                .await;
        }

        Ok(())
    }

    pub async fn download_artifact_data(&self, name: &str, dest: &Path) -> Result<u64> {
        let prefix = self.s3_prefix(&format!("artifacts/{}/data/", name));
        let artifact_data_dir = dest.join(name).join("data");
        fs::create_dir_all(&artifact_data_dir)?;

        let mut downloaded: u64 = 0;
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .context("Failed to list artifact data")?;

            for object in response.contents() {
                if let Some(key) = object.key() {
                    let relative_path = key.strip_prefix(&prefix).unwrap_or(key);
                    if relative_path.is_empty() {
                        continue;
                    }

                    let local_path = artifact_data_dir.join(relative_path);
                    if let Some(parent) = local_path.parent() {
                        fs::create_dir_all(parent)?;
                    }

                    let resp = self
                        .client
                        .get_object()
                        .bucket(&self.config.bucket)
                        .key(key)
                        .send()
                        .await?;

                    let body = resp.body.collect().await?.into_bytes();
                    downloaded += body.len() as u64;
                    fs::write(&local_path, &body)?;
                }
            }

            if response.is_truncated() == Some(true) {
                continuation_token = response.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(downloaded)
    }
}

fn merge_meta_bytes(local: &[u8], remote: &[u8]) -> Result<Vec<u8>> {
    let local_meta: MetaJson = serde_json::from_slice(local)?;
    let remote_meta: MetaJson = serde_json::from_slice(remote)?;

    let merged = if local_meta.status.as_deref() == Some("completed") {
        local_meta
    } else if remote_meta.status.as_deref() == Some("completed") {
        remote_meta
    } else {
        match (&local_meta.finished_at, &remote_meta.finished_at) {
            (Some(l), Some(r)) => {
                if l >= r {
                    local_meta
                } else {
                    remote_meta
                }
            }
            (Some(_), None) => local_meta,
            (None, Some(_)) => remote_meta,
            (None, None) => local_meta,
        }
    };

    Ok(serde_json::to_vec_pretty(&merged)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_s3::primitives::DateTime;

    fn object(secs_ago: i64, size: i64) -> Object {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        Object::builder()
            .key("runs/p/r/images/val/dets/step_1000.png")
            .last_modified(DateTime::from_secs(now - secs_ago))
            .size(size)
            .build()
    }

    #[test]
    fn run_upload_skips_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        for file in [
            "meta.json",
            "metrics/train/loss.csv",
            "checkpoints.json",
            "checkpoints/100/model.pt",
            "checkpoints/100/model.pt.part",
            "checkpoints/100/meta.json",
            "checkpoints_notes.txt",
        ] {
            let path = run_dir.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"x").unwrap();
        }

        let mut uploaded: Vec<String> = run_files_to_upload(run_dir)
            .unwrap()
            .iter()
            .map(|path| path.strip_prefix(run_dir).unwrap().display().to_string())
            .collect();
        uploaded.sort();

        assert_eq!(
            uploaded,
            [
                "checkpoints_notes.txt",
                "meta.json",
                "metrics/train/loss.csv"
            ]
        );
    }

    #[test]
    fn blob_is_current_when_older_remote_and_same_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("step_1000.png");
        fs::write(&path, b"12345").unwrap();

        assert!(local_is_current(&path, &object(60, 5), false));
    }

    #[test]
    fn blob_is_stale_when_size_differs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("step_1000.png");
        fs::write(&path, b"12345").unwrap();

        assert!(!local_is_current(&path, &object(60, 4), false));
    }

    #[test]
    fn mergeable_file_ignores_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("loss.csv");
        fs::write(&path, b"step,value\n1,0.5\n").unwrap();

        assert!(local_is_current(&path, &object(60, 4), true));
    }

    #[test]
    fn newer_or_same_second_remote_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("step_1000.png");
        fs::write(&path, b"12345").unwrap();

        assert!(!local_is_current(&path, &object(-60, 5), false));
        assert!(!local_is_current(&path, &object(0, 5), false));
        assert!(!local_is_current(&path, &object(0, 5), true));
    }

    #[test]
    fn missing_local_or_remote_metadata_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("step_1000.png");
        fs::write(&path, b"12345").unwrap();

        assert!(!local_is_current(
            &dir.path().join("missing.png"),
            &object(60, 5),
            false
        ));
        assert!(!local_is_current(dir.path(), &object(60, 5), false));
        let no_mtime = Object::builder().key("k").size(5).build();
        assert!(!local_is_current(&path, &no_mtime, false));
        let no_size = Object::builder()
            .key("k")
            .last_modified(DateTime::from_secs(0))
            .build();
        assert!(!local_is_current(&path, &no_size, false));
        assert!(local_is_current(&path, &no_size, true));
    }

    #[test]
    fn mergeable_keys() {
        assert!(is_mergeable("runs/p/r/metrics/train/loss.csv"));
        assert!(is_mergeable("runs/p/r/system.csv"));
        assert!(is_mergeable("runs/p/r/images/val/dets.jsonl"));
        assert!(is_mergeable("runs/p/r/meta.json"));
        assert!(!is_mergeable("runs/p/r/checkpoints.json"));
        assert!(!is_mergeable("runs/p/r/images/val/dets/step_1000.png"));
    }
}
