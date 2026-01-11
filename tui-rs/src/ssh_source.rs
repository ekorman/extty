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

use crate::data::{Example, MetricPoint, Run, RunStatus};

/// Global tokio runtime for SSH operations - reused across all calls
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().unwrap_or_else(|e| {
        eprintln!("Failed to create tokio runtime for SSH operations: {e}");
        std::process::exit(1);
    })
});

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
            // No username specified, use current user
            let username = std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .unwrap_or_else(|_| "root".to_string());
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
        // Accept all host keys (similar to ssh2's AutoAddPolicy)
        Ok(true)
    }
}

/// Execute a command and return stdout
async fn exec_command(handle: &mut Handle<Client>, cmd: &str) -> Result<String> {
    let mut channel = handle.channel_open_session().await?;
    channel.exec(true, cmd).await?;

    let mut output = String::new();
    let mut exit_status: Option<u32> = None;

    while let Some(msg) = channel.wait().await {
        match msg {
            russh::ChannelMsg::Data { ref data } => {
                output.push_str(&String::from_utf8_lossy(data));
            }
            russh::ChannelMsg::ExitStatus { exit_status: status } => {
                exit_status = Some(status);
                break;
            }
            _ => {}
        }
    }

    // Check if command succeeded
    match exit_status {
        Some(0) => Ok(output),
        Some(code) => Err(anyhow!("Command failed with exit code {}: {}", code, cmd)),
        None => Err(anyhow!("Command did not return exit status: {}", cmd)),
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
        let mut session = connect(config).await?;

        // List directories in ~/.ex/runs/
        let stdout = exec_command(
            &mut session,
            "cd ~/.ex/runs 2>/dev/null && ls -1 || true",
        )
        .await?;

        let mut runs = Vec::new();

        for line in stdout.lines() {
            let run_name = line.trim();
            if run_name.is_empty() {
                continue;
            }

            // Load each run using the same session
            if let Ok(run) = load_run_ssh_with_session(&mut session, run_name).await {
                runs.push(run);
            }
        }

        // Sort by name descending (newest first)
        runs.sort_by(|a, b| b.name.cmp(&a.name));
        Ok(runs)
    })
}

/// Load a single run from SSH (creates a new connection)
pub fn load_run_ssh(config: &SshConfig, run_name: &str) -> Result<Run> {
    RUNTIME.block_on(async {
        let mut session = connect(config).await?;
        load_run_ssh_with_session(&mut session, run_name).await
    })
}

/// Load a single run using an existing SSH session (internal helper)
async fn load_run_ssh_with_session(session: &mut Handle<Client>, run_name: &str) -> Result<Run> {
    let run_path = format!("~/.ex/runs/{}", run_name);

    // Check if meta.json exists (use || true to ensure exit code 0)
    let check_output = exec_command(
        session,
        &format!("test -f {}/meta.json && echo ok || true", run_path)
    ).await?;

    if !check_output.trim().contains("ok") {
        return Err(anyhow!("Run {} does not have meta.json", run_name));
    }

    // Load metadata
    let (start_time, end_time, status, config_json) =
        load_run_meta_ssh(session, &run_path).await?;

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
) -> Result<(
    Option<DateTime<Local>>,
    Option<DateTime<Local>>,
    RunStatus,
    Option<serde_json::Value>,
)> {
    let content = exec_command(session, &format!("cat {}/meta.json", run_path)).await?;

    let meta: RunMeta =
        serde_json::from_str(&content).context("Failed to parse meta.json")?;

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
            // Infer from end_time
            if end_time.is_some() {
                RunStatus::Completed
            } else {
                RunStatus::Running
            }
        }
    };

    Ok((start_time, end_time, status, meta.config))
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

        // Parse JSONL
        let examples = parse_examples_jsonl(&jsonl_content)?;

        // Example name is path without .jsonl extension
        let example_name = jsonl_path.strip_suffix(".jsonl").unwrap_or(jsonl_path);
        examples_map.insert(example_name.to_string(), examples);
    }

    Ok(examples_map)
}

fn parse_examples_jsonl(content: &str) -> Result<Vec<Example>> {
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let row: ExampleRow = serde_json::from_str(line)?;
        examples.push(Example {
            step: row.step,
            prompt: row.data.prompt,
            response: row.data.response,
        });
    }

    examples.sort_by_key(|e| e.step);
    Ok(examples)
}
