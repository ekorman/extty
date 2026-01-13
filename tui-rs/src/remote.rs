use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct RemoteSync {
    client: RemoteClient,
    runs_dir: PathBuf,
    cursors: HashMap<String, RunCursor>,
}

#[derive(Debug, Default)]
struct RunCursor {
    metric_step: u64,
    example_step: u64,
    system_step: u64,
}

#[derive(Debug)]
struct RemoteClient {
    base_url: String,
    client: Client,
    headers: HeaderMap,
}

impl RemoteSync {
    pub fn new(base_url: String, token: Option<String>, runs_dir: PathBuf) -> Result<Self> {
        let client = RemoteClient::new(base_url, token)?;
        Ok(Self {
            client,
            runs_dir,
            cursors: HashMap::new(),
        })
    }

    pub fn sync(&mut self) -> Result<()> {
        let runs = self.client.fetch_runs()?;
        for run in runs {
            self.ensure_run_dir(&run)?;
            let cursor = self.cursors.entry(run.name.clone()).or_default();

            let metrics = self.client.fetch_metrics(&run.name, cursor.metric_step)?;
            cursor.metric_step =
                write_metric_series(&self.runs_dir, &run.name, metrics, cursor.metric_step)?;

            let examples = self.client.fetch_examples(&run.name, cursor.example_step)?;
            cursor.example_step =
                write_example_series(&self.runs_dir, &run.name, examples, cursor.example_step)?;

            let system = self.client.fetch_system(&run.name, cursor.system_step)?;
            cursor.system_step =
                write_system_points(&self.runs_dir, &run.name, system, cursor.system_step)?;
        }
        Ok(())
    }

    fn ensure_run_dir(&self, run: &RunInfo) -> Result<()> {
        let run_dir = self.runs_dir.join(&run.name);
        if !run_dir.exists() {
            fs::create_dir_all(run_dir.join("metrics"))?;
            fs::create_dir_all(run_dir.join("examples"))?;
            write_meta(&run_dir, run)?;
        }
        Ok(())
    }
}

impl RemoteClient {
    fn new(base_url: String, token: Option<String>) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if let Some(token) = token {
            let value = HeaderValue::from_str(&format!("Bearer {}", token))
                .context("Invalid token for Authorization header")?;
            headers.insert(AUTHORIZATION, value);
        }
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
            headers,
        })
    }

    fn fetch_runs(&self) -> Result<Vec<RunInfo>> {
        let url = format!("{}/runs", self.base_url);
        let response = self
            .client
            .get(url)
            .headers(self.headers.clone())
            .send()?
            .error_for_status()?;
        let payload: RunsResponse = response.json()?;
        Ok(payload.runs)
    }

    fn fetch_metrics(&self, run_name: &str, step: u64) -> Result<MetricsResponse> {
        let url = format!("{}/runs/{}/metrics?step={}", self.base_url, run_name, step);
        let response = self
            .client
            .get(url)
            .headers(self.headers.clone())
            .send()?
            .error_for_status()?;
        Ok(response.json()?)
    }

    fn fetch_examples(&self, run_name: &str, step: u64) -> Result<ExamplesResponse> {
        let url = format!("{}/runs/{}/examples?step={}", self.base_url, run_name, step);
        let response = self
            .client
            .get(url)
            .headers(self.headers.clone())
            .send()?
            .error_for_status()?;
        Ok(response.json()?)
    }

    fn fetch_system(&self, run_name: &str, step: u64) -> Result<SystemResponse> {
        let url = format!("{}/runs/{}/system?step={}", self.base_url, run_name, step);
        let response = self
            .client
            .get(url)
            .headers(self.headers.clone())
            .send()?
            .error_for_status()?;
        Ok(response.json()?)
    }
}

#[derive(Debug, Deserialize)]
struct RunsResponse {
    runs: Vec<RunInfo>,
}

#[derive(Debug, Deserialize)]
struct RunInfo {
    name: String,
    project: String,
    config: serde_json::Value,
    started_at: String,
    finished_at: Option<String>,
    status: String,
}

#[derive(Debug, Deserialize)]
struct MetricPoint {
    step: u64,
    timestamp: f64,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct MetricSeries {
    name: String,
    points: Vec<MetricPoint>,
}

#[derive(Debug, Deserialize)]
struct MetricsResponse {
    metrics: Vec<MetricSeries>,
}

#[derive(Debug, Deserialize)]
struct ExampleRecord {
    step: u64,
    timestamp: f64,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ExampleSeries {
    name: String,
    records: Vec<ExampleRecord>,
}

#[derive(Debug, Deserialize)]
struct ExamplesResponse {
    examples: Vec<ExampleSeries>,
}

#[derive(Debug, Deserialize)]
struct SystemPoint {
    step: u64,
    timestamp: f64,
    ram_used_gb: f64,
    ram_total_gb: f64,
    gpu_mem_used_gb: Option<f64>,
    gpu_mem_total_gb: Option<f64>,
    gpu_util_pct: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SystemResponse {
    points: Vec<SystemPoint>,
}

fn write_metric_series(
    runs_dir: &Path,
    run_name: &str,
    response: MetricsResponse,
    current_step: u64,
) -> Result<u64> {
    let mut max_step = current_step;
    for series in response.metrics {
        if series.points.is_empty() {
            continue;
        }
        let path = metric_path(runs_dir, run_name, &series.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        if file.metadata()?.len() == 0 {
            writeln!(file, "step,timestamp,value")?;
        }
        for point in series.points {
            max_step = max_step.max(point.step);
            writeln!(
                file,
                "{},{:.6},{}",
                point.step, point.timestamp, point.value
            )?;
        }
    }
    Ok(max_step)
}

fn write_example_series(
    runs_dir: &Path,
    run_name: &str,
    response: ExamplesResponse,
    current_step: u64,
) -> Result<u64> {
    let mut max_step = current_step;
    for series in response.examples {
        if series.records.is_empty() {
            continue;
        }
        let path = example_path(runs_dir, run_name, &series.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        for record in series.records {
            max_step = max_step.max(record.step);
            let row = serde_json::json!({
                "step": record.step,
                "timestamp": record.timestamp,
                "data": record.data,
            });
            writeln!(file, "{}", row)?;
        }
    }
    Ok(max_step)
}

fn write_system_points(
    runs_dir: &Path,
    run_name: &str,
    response: SystemResponse,
    current_step: u64,
) -> Result<u64> {
    let mut max_step = current_step;
    if response.points.is_empty() {
        return Ok(max_step);
    }
    let path = runs_dir.join(run_name).join("system.csv");
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    if file.metadata()?.len() == 0 {
        writeln!(
            file,
            "timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct"
        )?;
    }
    for point in response.points {
        max_step = max_step.max(point.step);
        let gpu_mem_used = point
            .gpu_mem_used_gb
            .map(|v| format!("{:.2}", v))
            .unwrap_or_default();
        let gpu_mem_total = point
            .gpu_mem_total_gb
            .map(|v| format!("{:.2}", v))
            .unwrap_or_default();
        let gpu_util = point
            .gpu_util_pct
            .map(|v| format!("{:.1}", v))
            .unwrap_or_default();
        writeln!(
            file,
            "{:.6},{:.2},{:.2},{},{},{}",
            point.timestamp,
            point.ram_used_gb,
            point.ram_total_gb,
            gpu_mem_used,
            gpu_mem_total,
            gpu_util
        )?;
    }
    Ok(max_step)
}

fn write_meta(run_dir: &Path, run: &RunInfo) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Meta<'a> {
        project: &'a str,
        run_name: &'a str,
        config: serde_json::Value,
        started_at: &'a str,
        finished_at: Option<String>,
        status: &'a str,
    }

    let meta = Meta {
        project: &run.project,
        run_name: &run.name,
        config: run.config.clone(),
        started_at: &run.started_at,
        finished_at: run.finished_at.clone(),
        status: &run.status,
    };
    let path = run_dir.join("meta.json");
    let content = serde_json::to_vec_pretty(&meta)?;
    fs::write(path, content)?;
    Ok(())
}

fn metric_path(runs_dir: &Path, run_name: &str, metric: &str) -> PathBuf {
    let safe = sanitize_metric_name(metric);
    runs_dir
        .join(run_name)
        .join("metrics")
        .join(format!("{}.csv", safe))
}

fn example_path(runs_dir: &Path, run_name: &str, group: &str) -> PathBuf {
    let safe = sanitize_metric_name(group);
    runs_dir
        .join(run_name)
        .join("examples")
        .join(format!("{}.jsonl", safe))
}

fn sanitize_metric_name(name: &str) -> String {
    let mut sanitized = String::new();
    for (i, part) in name.split('/').enumerate() {
        if i > 0 {
            sanitized.push('/');
        }
        for ch in part.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                sanitized.push(ch);
            } else {
                sanitized.push('_');
            }
        }
    }
    sanitized
}
