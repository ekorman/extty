use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Local};
use once_cell::sync::Lazy;
use russh::client::{self, Handle, Handler};
use russh::keys::key;
use russh_keys::load_secret_key;
use serde::Deserialize;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use crate::data::{Example, MetricPoint, Run, RunStatus};

/// Global tokio runtime for SSH operations - reused across all calls
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().unwrap_or_else(|e| {
        eprintln!("Failed to create tokio runtime for SSH operations: {e}");
        std::process::exit(1);
    })
});

/// Cached SSH session - reused across calls to avoid reconnection overhead
static CACHED_SESSION: Lazy<Mutex<Option<(String, Handle<Client>)>>> =
    Lazy::new(|| Mutex::new(None));

#[derive(Debug, Deserialize)]
struct RunMeta {
    #[allow(dead_code)]
    project: Option<String>,
    #[allow(dead_code)]
    run_name: Option<String>,
    started_at: Option<String>,
    finished_at: Option<String>,
    status: Option<String>,
    config: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct MetricRow {
    step: u64,
    #[allow(dead_code)]
    timestamp: f64,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct ExampleRow {
    step: u64,
    #[allow(dead_code)]
    timestamp: f64,
    data: ExampleData,
}

#[derive(Debug, Deserialize)]
struct ExampleData {
    prompt: String,
    response: String,
}

/// SSH connection configuration
#[derive(Clone)]
pub struct SshConfig {
    pub username: String,
    pub hostname: String,
    pub port: u16,
}

impl SshConfig {
    /// Parse user@host format
    pub fn parse(server: &str) -> Result<Self> {
        let parts: Vec<&str> = server.split('@').collect();

        let (username, hostname) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else if parts.len() == 1 {
            // No username specified, try to get current user
            let username = std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .map_err(|_| anyhow!(
                    "Cannot determine username. Please specify in format: user@host"
                ))?;
            (username, parts[0].to_string())
        } else {
            return Err(anyhow!("Invalid server format: {}", server));
        };

        Ok(SshConfig {
            username,
            hostname,
            port: 22,
        })
    }
}

/// SSH client handler
struct Client;

#[async_trait::async_trait]
impl Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // WARNING: accepting all host keys without verification is insecure and
        // makes the connection susceptible to man-in-the-middle attacks.
        // Consider implementing proper host key verification against known_hosts
        // before using this in production.
        eprintln!(
            "Warning: SSH host key verification is disabled; accepting server key without checking known_hosts."
        );
        Ok(true)
    }
}

/// Execute a command and return stdout
async fn exec_command(handle: &mut Handle<Client>, cmd: &str) -> Result<String> {
    let mut channel = handle.channel_open_session().await?;
    channel.exec(true, cmd).await?;

    let mut output = String::new();
    let mut exit_status: Option<u32> = None;
    let mut got_eof = false;

    // Read until we have both EOF and ExitStatus, or channel closes
    while let Some(msg) = channel.wait().await {
        match msg {
            russh::ChannelMsg::Data { ref data } => {
                output.push_str(&String::from_utf8_lossy(data));
            }
            russh::ChannelMsg::ExitStatus { exit_status: status } => {
                exit_status = Some(status);
                if got_eof {
                    break;
                }
            }
            russh::ChannelMsg::Eof => {
                got_eof = true;
                if exit_status.is_some() {
                    break;
                }
            }
            _ => {}
        }
    }

    match exit_status {
        Some(0) => Ok(output),
        Some(code) => Err(anyhow!("Command failed with exit code {}: {}", code, cmd)),
        None => Ok(output), // No exit status but got data - assume success
    }
}

/// Create an SSH session
async fn connect(config: &SshConfig) -> Result<Handle<Client>> {
    let client_config = Arc::new(client::Config::default());
    let mut session = client::connect(client_config, (&*config.hostname, config.port), Client)
        .await
        .context("Failed to connect to SSH server")?;

    // Try to load SSH keys from standard locations
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not find home directory"))?;
    let ssh_dir = home.join(".ssh");

    // Try common key files
    let key_files = vec!["id_rsa", "id_ed25519", "id_ecdsa"];
    let mut tried_keys = Vec::new();

    for key_file in key_files {
        let key_path = ssh_dir.join(key_file);
        if !key_path.exists() {
            continue;
        }

        tried_keys.push(key_file.to_string());

        // First try without password
        match load_secret_key(&key_path, None) {
            Ok(key_pair) => {
                let auth_result = session
                    .authenticate_publickey(&config.username, Arc::new(key_pair))
                    .await;

                if let Ok(true) = auth_result {
                    return Ok(session);
                }
                // Key loaded but auth failed - continue to next key
            }
            Err(e) => {
                // Check if error suggests the key is encrypted
                let error_msg = e.to_string().to_lowercase();
                if error_msg.contains("encrypted") || error_msg.contains("password") {
                    // Prompt for password
                    eprintln!("Key {} is encrypted.", key_file);
                    match rpassword::prompt_password(format!("Enter passphrase for {}: ", key_file)) {
                        Ok(password) => {
                            // Try loading with password
                            if let Ok(key_pair) = load_secret_key(&key_path, Some(&password)) {
                                let auth_result = session
                                    .authenticate_publickey(&config.username, Arc::new(key_pair))
                                    .await;

                                if let Ok(true) = auth_result {
                                    return Ok(session);
                                }
                            }
                            // Password was wrong or auth failed - continue to next key
                        }
                        Err(_) => {
                            // Failed to read password - skip this key
                            continue;
                        }
                    }
                }
                // Otherwise key load failed for other reasons - continue to next key
            }
        }
    }

    // Provide informative error message
    if !tried_keys.is_empty() {
        Err(anyhow!(
            "SSH authentication failed. Tried keys: {} in ~/.ssh/. \
             Keys may have wrong permissions, wrong password, or not be authorized on server.",
            tried_keys.join(", ")
        ))
    } else {
        Err(anyhow!(
            "SSH authentication failed. No SSH keys found in ~/.ssh/. \
             Expected to find id_rsa, id_ed25519, or id_ecdsa."
        ))
    }
}

/// List all runs from a remote server via SSH
pub fn load_runs_ssh(config: &SshConfig) -> Result<Vec<Run>> {
    RUNTIME.block_on(async {
        ensure_session(config).await?;

        let mut cache = CACHED_SESSION.lock().await;
        let (_key, session) = cache.as_mut().unwrap();

        // List directories in ~/.ex/runs/
        let stdout = exec_command(
            session,
            "cd \"$HOME/.ex/runs\" 2>/dev/null && ls -1 || true",
        )
        .await?;

        let mut runs = Vec::new();

        for line in stdout.lines() {
            let run_name = line.trim();
            if run_name.is_empty() {
                continue;
            }

            // Load each run using the same session
            match load_run_ssh_with_session(session, run_name).await {
                Ok(run) => runs.push(run),
                Err(e) => {
                    // Log error but continue loading other runs
                    eprintln!("Warning: Failed to load run '{}': {}", run_name, e);
                }
            }
        }

        // Sort by name descending (newest first)
        runs.sort_by(|a, b| b.name.cmp(&a.name));
        Ok(runs)
    })
}

/// Load a single run from SSH (reuses cached connection)
pub fn load_run_ssh(config: &SshConfig, run_name: &str) -> Result<Run> {
    RUNTIME.block_on(async {
        ensure_session(config).await?;

        let mut cache = CACHED_SESSION.lock().await;
        let (_key, session) = cache.as_mut().unwrap();

        load_run_ssh_with_session(session, run_name).await
    })
}

/// Ensure we have a valid cached session for this config
async fn ensure_session(config: &SshConfig) -> Result<()> {
    let cache_key = format!("{}@{}:{}", config.username, config.hostname, config.port);

    let mut cache = CACHED_SESSION.lock().await;

    // Check if we need a new session
    let needs_new_session = if let Some((cached_key, session)) = cache.as_mut() {
        if cached_key != &cache_key {
            // Different server
            true
        } else {
            // Same server - test if session is still alive.
            // Use a simple shell builtin that should always succeed if the
            // session is healthy, minimizing dependence on remote shell setup.
            exec_command(session, "true").await.is_err()
        }
    } else {
        // No cached session
        true
    };

    if needs_new_session {
        // Need to drop the lock before calling connect (which might prompt for password)
        drop(cache);

        // Create new connection
        let session = connect(config).await?;

        // Re-acquire lock and store session
        let mut cache = CACHED_SESSION.lock().await;
        *cache = Some((cache_key, session));
    }

    Ok(())
}

/// Load a single run using an existing SSH session (internal helper)
async fn load_run_ssh_with_session(session: &mut Handle<Client>, run_name: &str) -> Result<Run> {
    // Validate run_name to prevent path traversal or injection of path separators.
    // We only expect simple directory names here.
    if run_name.contains('/') || run_name.contains('\\') || run_name.contains("..") {
        return Err(anyhow!("Invalid run name: {}", run_name));
    }
    let run_path = format!("~/.ex/runs/{}", run_name);

    // Check if meta.json exists (use || true to ensure exit code 0)
    let check_output = exec_command(
        session,
        &format!("test -f {}/meta.json && echo ok || true", run_path)
    ).await?;

    if check_output.trim() != "ok" {
        return Err(anyhow!("Run {} does not have meta.json", run_name));
    }

    // Load metadata (tolerates partial writes)
    let (start_time, end_time, status, config_json) =
        load_run_meta_ssh(session, &run_path).await;

    // Load metrics
    let metrics = load_metrics_ssh(session, &run_path).await?;

    // Load examples
    let examples = load_examples_ssh(session, &run_path).await?;

    Ok(Run {
        name: run_name.to_string(),
        path: PathBuf::from(&run_path),
        metrics,
        examples,
        start_time,
        end_time,
        status,
        config: config_json,
    })
}

async fn load_run_meta_ssh(
    session: &mut Handle<Client>,
    run_path: &str,
) -> (
    Option<DateTime<Local>>,
    Option<DateTime<Local>>,
    RunStatus,
    Option<serde_json::Value>,
) {
    let Ok(content) = exec_command(session, &format!("cat {}/meta.json", run_path)).await else {
        return (None, None, RunStatus::Unknown, None);
    };

    let Ok(meta) = serde_json::from_str::<RunMeta>(&content) else {
        return (None, None, RunStatus::Unknown, None);
    };

    let start_time = meta
        .started_at
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Local));

    let end_time = meta
        .finished_at
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Local));

    let status = match meta.status.as_deref() {
        Some("completed") => RunStatus::Completed,
        Some("running") => RunStatus::Running,
        _ => {
            if end_time.is_some() {
                RunStatus::Completed
            } else {
                RunStatus::Running
            }
        }
    };

    (start_time, end_time, status, meta.config)
}

async fn load_metrics_ssh(
    session: &mut Handle<Client>,
    run_path: &str,
) -> Result<HashMap<String, Vec<MetricPoint>>> {
    let mut metrics = HashMap::new();

    // List all CSV files in metrics directory
    let stdout = exec_command(
        session,
        &format!(
            "cd {}/metrics 2>/dev/null && find . -name '*.csv' -type f || true",
            run_path
        ),
    )
    .await?;

    if stdout.trim().is_empty() {
        return Ok(metrics);
    }

    for line in stdout.lines() {
        let csv_path = line.trim().trim_start_matches("./");
        if csv_path.is_empty() {
            continue;
        }

        // Read the CSV file
        let csv_content = exec_command(
            session,
            &format!("cat {}/metrics/{}", run_path, csv_path),
        )
        .await?;

        // Parse CSV
        let points = parse_metric_csv(&csv_content)?;

        // Metric name is path without .csv extension
        let metric_name = csv_path.strip_suffix(".csv").unwrap_or(csv_path);
        metrics.insert(metric_name.to_string(), points);
    }

    Ok(metrics)
}

fn parse_metric_csv(content: &str) -> Result<Vec<MetricPoint>> {
    let mut reader = csv::Reader::from_reader(content.as_bytes());
    let mut points = Vec::new();

    for result in reader.deserialize() {
        let row: MetricRow = result?;
        points.push(MetricPoint {
            step: row.step,
            value: row.value,
        });
    }

    points.sort_by_key(|p| p.step);
    Ok(points)
}

async fn load_examples_ssh(
    session: &mut Handle<Client>,
    run_path: &str,
) -> Result<HashMap<String, Vec<Example>>> {
    let mut examples_map = HashMap::new();

    // List all JSONL files in examples directory
    let stdout = exec_command(
        session,
        &format!(
            "cd {}/examples 2>/dev/null && find . -name '*.jsonl' -type f || true",
            run_path
        ),
    )
    .await?;

    if stdout.trim().is_empty() {
        return Ok(examples_map);
    }

    for line in stdout.lines() {
        let jsonl_path = line.trim().trim_start_matches("./");
        if jsonl_path.is_empty() {
            continue;
        }

        // Read the JSONL file
        let jsonl_content = exec_command(
            session,
            &format!("cat {}/examples/{}", run_path, jsonl_path),
        )
        .await?;

        // Parse JSONL (silently skip malformed lines)
        let examples = parse_examples_jsonl(&jsonl_content);

        // Example name is path without .jsonl extension
        let example_name = jsonl_path.strip_suffix(".jsonl").unwrap_or(jsonl_path);
        examples_map.insert(example_name.to_string(), examples);
    }

    Ok(examples_map)
}

fn parse_examples_jsonl(content: &str) -> Vec<Example> {
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(row) = serde_json::from_str::<ExampleRow>(line) {
            examples.push(Example {
                step: row.step,
                prompt: row.data.prompt,
                response: row.data.response,
            });
        }
    }

    examples.sort_by_key(|e| e.step);
    examples
}
