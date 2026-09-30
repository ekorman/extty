use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::checkpoints::{self, Status};
use crate::data::artifacts_dir;
use crate::s3;

pub struct PruneLocalOptions {
    pub target: Option<String>,
    pub artifact: Option<String>,
    pub no_checkpoints: bool,
    pub no_artifacts: bool,
    pub dry_run: bool,
    pub assume_yes: bool,
}

struct CheckpointCandidate {
    project: String,
    run_name: String,
    step: u64,
    local_path: PathBuf,
    size_bytes: u64,
    /// `None` when S3 could not be checked.
    status: Option<Status>,
}

impl CheckpointCandidate {
    fn deletable(&self) -> bool {
        self.status.is_some_and(Status::local_is_deletable)
    }
}

/// A run's project, name, and local checkpoint directories keyed by step.
type LocalCheckpointRun = (String, String, Vec<(u64, PathBuf)>);

struct ArtifactCandidate {
    name: String,
    local_dir: PathBuf,
    size_bytes: u64,
    in_s3: bool,
}

pub fn run_local(opts: PruneLocalOptions) -> Result<()> {
    if opts.no_checkpoints && opts.no_artifacts {
        println!("Nothing to do: --no-checkpoints and --no-artifacts both set.");
        return Ok(());
    }

    let config = s3::load_config()?.ok_or_else(|| {
        anyhow::anyhow!(
            "S3 not configured. Create {} with:\n\n\
             bucket = \"your-bucket\"\n\
             region = \"us-west-2\"\n",
            s3::config_path().display()
        )
    })?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let client = s3::S3Client::new(config).await?;

        let checkpoints = if opts.no_checkpoints {
            Vec::new()
        } else {
            let (project_filter, run_filter) = parse_target(&opts.target);
            let runs =
                find_local_checkpoint_runs(project_filter.as_deref(), run_filter.as_deref())?;
            verify_checkpoints(&client, runs).await?
        };

        let artifacts = if opts.no_artifacts {
            Vec::new()
        } else {
            let locals = find_local_artifacts(opts.artifact.as_deref())?;
            verify_artifacts(&client, locals).await?
        };

        print_checkpoint_summary(&checkpoints);
        print_artifact_summary(&artifacts);

        let safe_ckpts: Vec<&CheckpointCandidate> =
            checkpoints.iter().filter(|c| c.deletable()).collect();
        let safe_arts: Vec<&ArtifactCandidate> = artifacts.iter().filter(|a| a.in_s3).collect();

        let total_bytes: u64 = safe_ckpts.iter().map(|c| c.size_bytes).sum::<u64>()
            + safe_arts.iter().map(|a| a.size_bytes).sum::<u64>();
        let total_count = safe_ckpts.len() + safe_arts.len();

        if total_count == 0 {
            println!("Nothing to delete.");
            return Ok(());
        }

        println!(
            "\nWill delete {} items, freeing {}.",
            total_count,
            format_bytes(total_bytes)
        );

        if opts.dry_run {
            println!("(dry-run) no files removed.");
            return Ok(());
        }

        if !opts.assume_yes
            && !confirm(&format!(
                "Delete {} items ({})? [y/N] ",
                total_count,
                format_bytes(total_bytes)
            ))?
        {
            println!("Aborted.");
            return Ok(());
        }

        for c in &safe_ckpts {
            delete_checkpoint_dir(c)?;
        }
        for a in &safe_arts {
            delete_artifact_dir(a)?;
        }

        println!("Freed {}.", format_bytes(total_bytes));
        Ok::<(), anyhow::Error>(())
    })
}

fn parse_target(target: &Option<String>) -> (Option<String>, Option<String>) {
    match target {
        None => (None, None),
        Some(t) => {
            if t.ends_with('/') {
                (Some(t.trim_end_matches('/').to_string()), None)
            } else if t.contains('/') {
                let (proj, run) = t.split_once('/').unwrap();
                (Some(proj.to_string()), Some(run.to_string()))
            } else {
                (Some(t.clone()), None)
            }
        }
    }
}

fn runs_dir() -> PathBuf {
    crate::paths::extty_home().join("runs")
}

fn find_local_checkpoint_runs(
    project_filter: Option<&str>,
    run_filter: Option<&str>,
) -> Result<Vec<LocalCheckpointRun>> {
    let runs_dir = runs_dir();
    if !runs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let mut proj_entries: Vec<_> = fs::read_dir(&runs_dir)?.filter_map(|e| e.ok()).collect();
    proj_entries.sort_by_key(|e| e.file_name());

    for proj_entry in proj_entries {
        let proj_path = proj_entry.path();
        if !proj_path.is_dir() {
            continue;
        }
        let proj_name = proj_entry.file_name().to_string_lossy().to_string();
        if let Some(f) = project_filter
            && proj_name != f
        {
            continue;
        }

        let mut run_entries: Vec<_> = fs::read_dir(&proj_path)?.filter_map(|e| e.ok()).collect();
        run_entries.sort_by_key(|e| e.file_name());

        for run_entry in run_entries {
            let run_path = run_entry.path();
            if !run_path.is_dir() {
                continue;
            }
            let run_name = run_entry.file_name().to_string_lossy().to_string();
            if let Some(f) = run_filter
                && run_name != f
            {
                continue;
            }

            let step_dirs = checkpoints::local_step_dirs(&run_path);
            if !step_dirs.is_empty() {
                out.push((proj_name.clone(), run_name, step_dirs));
            }
        }
    }
    Ok(out)
}

async fn verify_checkpoints(
    client: &s3::S3Client,
    runs: Vec<LocalCheckpointRun>,
) -> Result<Vec<CheckpointCandidate>> {
    let mut out = Vec::new();
    for (project, run_name, step_dirs) in runs {
        for (step, step_dir) in step_dirs {
            let local = checkpoints::read_local_copy(&step_dir);
            let status = match client
                .remote_checkpoint_meta(&project, &run_name, step)
                .await
            {
                Ok(remote) => checkpoints::status(local.as_ref(), remote.as_ref()),
                Err(e) => {
                    eprintln!(
                        "warning: could not check S3 for {}/{} step {}: {}",
                        project, run_name, step, e
                    );
                    None
                }
            };
            out.push(CheckpointCandidate {
                project: project.clone(),
                run_name: run_name.clone(),
                step,
                size_bytes: dir_size(&step_dir),
                local_path: step_dir,
                status,
            });
        }
    }
    Ok(out)
}

fn find_local_artifacts(name_filter: Option<&str>) -> Result<Vec<(String, PathBuf)>> {
    let dir = artifacts_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<_> = fs::read_dir(&dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    let mut out = Vec::new();
    for entry in entries {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(f) = name_filter
            && name != f
        {
            continue;
        }
        if !path.join("data").is_dir() {
            continue;
        }
        out.push((name, path));
    }
    Ok(out)
}

async fn verify_artifacts(
    client: &s3::S3Client,
    artifacts: Vec<(String, PathBuf)>,
) -> Result<Vec<ArtifactCandidate>> {
    let remote: HashSet<String> = match client.list_artifacts().await {
        Ok(list) => list.into_iter().map(|a| a.name).collect(),
        Err(e) => {
            eprintln!("warning: could not list S3 artifacts: {}", e);
            HashSet::new()
        }
    };

    Ok(artifacts
        .into_iter()
        .map(|(name, local_dir)| {
            let size_bytes = dir_size(&local_dir);
            let in_s3 = remote.contains(&name);
            ArtifactCandidate {
                name,
                local_dir,
                size_bytes,
                in_s3,
            }
        })
        .collect())
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_symlink() {
            continue;
        } else if p.is_dir() {
            total += dir_size(&p);
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

fn format_bytes(n: u64) -> String {
    let mut size = n as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"] {
        if size < 1024.0 {
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PiB", size)
}

fn print_checkpoint_summary(candidates: &[CheckpointCandidate]) {
    if candidates.is_empty() {
        return;
    }
    println!("Checkpoints:");
    let mut current_run: Option<(String, String)> = None;
    let mut sorted: Vec<&CheckpointCandidate> = candidates.iter().collect();
    sorted.sort_by(|a, b| {
        (a.project.as_str(), a.run_name.as_str(), a.step).cmp(&(
            b.project.as_str(),
            b.run_name.as_str(),
            b.step,
        ))
    });
    for c in sorted {
        let key = (c.project.clone(), c.run_name.clone());
        if current_run.as_ref() != Some(&key) {
            println!("  {}/{}", c.project, c.run_name);
            current_run = Some(key);
        }
        let mark = match c.status {
            Some(Status::Synced) => "✓ in S3",
            Some(Status::Cached) => "✓ in S3 (old download)",
            Some(Status::Diverged) => "✗ S3 has a different save — skip",
            Some(Status::LocalOnly | Status::Untracked | Status::RemoteOnly) => {
                "✗ not in S3 — skip"
            }
            None => "✗ could not check S3 — skip",
        };
        println!(
            "    step {:<10} {:<22} {}",
            c.step,
            mark,
            format_bytes(c.size_bytes)
        );
    }
}

fn print_artifact_summary(candidates: &[ArtifactCandidate]) {
    if candidates.is_empty() {
        return;
    }
    println!("Artifacts:");
    for c in candidates {
        let mark = if c.in_s3 {
            "✓ in S3"
        } else {
            "✗ not in S3 — skip"
        };
        println!(
            "  {:<60} {:<22} {}",
            c.name,
            mark,
            format_bytes(c.size_bytes)
        );
    }
}

fn confirm(prompt: &str) -> Result<bool> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let answer = input.trim().to_lowercase();
    Ok(answer == "y" || answer == "yes")
}

fn delete_checkpoint_dir(c: &CheckpointCandidate) -> Result<()> {
    fs::remove_dir_all(&c.local_path)
        .with_context(|| format!("Failed to delete {}", c.local_path.display()))
}

fn delete_artifact_dir(a: &ArtifactCandidate) -> Result<()> {
    let data_dir = a.local_dir.join("data");
    if data_dir.exists() {
        fs::remove_dir_all(&data_dir)
            .with_context(|| format!("Failed to delete {}", data_dir.display()))?;
    }
    let meta = a.local_dir.join("meta.json");
    if meta.exists() {
        fs::remove_file(&meta).ok();
    }
    if a.local_dir.exists() && fs::read_dir(&a.local_dir)?.next().is_none() {
        fs::remove_dir(&a.local_dir).ok();
    }
    Ok(())
}
