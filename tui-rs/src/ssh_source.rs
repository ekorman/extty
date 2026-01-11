use std::collections::HashMap;
use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Local};
use serde::Deserialize;
use ssh2::Session;

use crate::data::{Example, MetricPoint, Run, RunStatus};

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

    /// Create an SSH session
    pub fn connect(&self) -> Result<Session> {
        let tcp = TcpStream::connect(format!("{}:{}", self.hostname, self.port))
            .context("Failed to connect via TCP")?;

        let mut sess = Session::new()?;

        sess.set_tcp_stream(tcp);
        sess.handshake()
            .context("SSH handshake failed")?;

        // Try to authenticate using SSH agent
        sess.userauth_agent(&self.username)
            .context("SSH authentication failed")?;

        if !sess.authenticated() {
            return Err(anyhow!("SSH authentication failed"));
        }

        Ok(sess)
    }
}

/// List all runs from a remote server via SSH
pub fn load_runs_ssh(config: &SshConfig) -> Result<Vec<Run>> {
    let sess = config.connect()?;

    // List directories in ~/.ex/runs/
    let (stdout, _exit_code) = exec_command(
        &sess,
        "cd ~/.ex/runs 2>/dev/null && ls -1 || true"
    )?;

    let mut runs = Vec::new();

    for line in stdout.lines() {
        let run_name = line.trim();
        if run_name.is_empty() {
            continue;
        }

        // Load each run
        if let Ok(run) = load_run_ssh(config, run_name) {
            runs.push(run);
        }
    }

    // Sort by name descending (newest first)
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    Ok(runs)
}

/// Load a single run from SSH
pub fn load_run_ssh(config: &SshConfig, run_name: &str) -> Result<Run> {
    let sess = config.connect()?;
    let run_path = format!("~/.ex/runs/{}", run_name);

    // Check if meta.json exists
    let (_, exit_code) = exec_command(
        &sess,
        &format!("test -f {}/meta.json", run_path)
    )?;

    if exit_code != 0 {
        return Err(anyhow!("Run {} does not have meta.json", run_name));
    }

    // Load metadata
    let (start_time, end_time, status, config_json) = load_run_meta_ssh(&sess, &run_path)?;

    // Load metrics
    let metrics = load_metrics_ssh(&sess, &run_path)?;

    // Load examples
    let examples = load_examples_ssh(&sess, &run_path)?;

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

fn load_run_meta_ssh(
    sess: &Session,
    run_path: &str,
) -> Result<(Option<DateTime<Local>>, Option<DateTime<Local>>, RunStatus, Option<serde_json::Value>)> {
    let (content, _) = exec_command(sess, &format!("cat {}/meta.json", run_path))?;

    let meta: RunMeta = serde_json::from_str(&content)
        .context("Failed to parse meta.json")?;

    let start_time = meta.started_at
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Local));

    let end_time = meta.finished_at
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

fn load_metrics_ssh(sess: &Session, run_path: &str) -> Result<HashMap<String, Vec<MetricPoint>>> {
    let mut metrics = HashMap::new();

    // List all CSV files in metrics directory
    let (stdout, exit_code) = exec_command(
        sess,
        &format!("cd {}/metrics 2>/dev/null && find . -name '*.csv' -type f || true", run_path)
    )?;

    if exit_code != 0 || stdout.trim().is_empty() {
        return Ok(metrics);
    }

    for line in stdout.lines() {
        let csv_path = line.trim().trim_start_matches("./");
        if csv_path.is_empty() {
            continue;
        }

        // Read the CSV file
        let (csv_content, _) = exec_command(
            sess,
            &format!("cat {}/metrics/{}", run_path, csv_path)
        )?;

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

fn load_examples_ssh(sess: &Session, run_path: &str) -> Result<HashMap<String, Vec<Example>>> {
    let mut examples_map = HashMap::new();

    // List all JSONL files in examples directory
    let (stdout, exit_code) = exec_command(
        sess,
        &format!("cd {}/examples 2>/dev/null && find . -name '*.jsonl' -type f || true", run_path)
    )?;

    if exit_code != 0 || stdout.trim().is_empty() {
        return Ok(examples_map);
    }

    for line in stdout.lines() {
        let jsonl_path = line.trim().trim_start_matches("./");
        if jsonl_path.is_empty() {
            continue;
        }

        // Read the JSONL file
        let (jsonl_content, _) = exec_command(
            sess,
            &format!("cat {}/examples/{}", run_path, jsonl_path)
        )?;

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

/// Execute a command over SSH and return (stdout, exit_code)
fn exec_command(sess: &Session, cmd: &str) -> Result<(String, i32)> {
    let mut channel = sess.channel_session()
        .context("Failed to open SSH channel")?;

    channel.exec(cmd)
        .context("Failed to execute command")?;

    let mut stdout = String::new();
    channel.read_to_string(&mut stdout)
        .context("Failed to read stdout")?;

    channel.wait_close()
        .context("Failed to close channel")?;

    let exit_code = channel.exit_status()?;

    Ok((stdout, exit_code))
}
