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
    base_url: String,
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
        let client = RemoteClient::new(base_url.clone(), token)?;
        Ok(Self {
            client,
            runs_dir,
            cursors: HashMap::new(),
            base_url,
        })
    }

    pub fn sync(&mut self) -> Result<()> {
        let runs = self.client.fetch_runs()?;
        for run in runs {
            self.ensure_run_dir(&run)?;
            let run_dir = self.run_dir(&run);
            let cursor = self
                .cursors
                .entry(run.name.clone())
                .or_insert_with(|| read_cursor_from_disk(&run_dir));

            let metrics = self.client.fetch_metrics(&run.name, cursor.metric_step)?;
            cursor.metric_step = write_metric_series(&run_dir, metrics, cursor.metric_step)?;

            let examples = self.client.fetch_examples(&run.name, cursor.example_step)?;
            cursor.example_step = write_example_series(&run_dir, examples, cursor.example_step)?;

            let system = self.client.fetch_system(&run.name, cursor.system_step)?;
            cursor.system_step = write_system_points(&run_dir, system, cursor.system_step)?;

            // Update meta.json if the run has finished
            if run.status != "running" {
                update_meta(&run_dir, &run)?;
            }
        }
        Ok(())
    }

    fn ensure_run_dir(&self, run: &RunInfo) -> Result<()> {
        let project_dir = if run.project.is_empty() {
            "_default"
        } else {
            &run.project
        };
        let run_dir = self.runs_dir.join(project_dir).join(&run.name);
        if !run_dir.exists() {
            fs::create_dir_all(run_dir.join("metrics"))?;
            fs::create_dir_all(run_dir.join("examples"))?;
            write_meta(&run_dir, run, &self.base_url)?;
        }
        Ok(())
    }

    fn run_dir(&self, run: &RunInfo) -> PathBuf {
        let project_dir = if run.project.is_empty() {
            "_default"
        } else {
            &run.project
        };
        self.runs_dir.join(project_dir).join(&run.name)
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
    run_dir: &Path,
    response: MetricsResponse,
    current_step: u64,
) -> Result<u64> {
    let mut max_step = current_step;
    for series in response.metrics {
        if series.points.is_empty() {
            continue;
        }
        let path = metric_path(run_dir, &series.name);
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
    run_dir: &Path,
    response: ExamplesResponse,
    current_step: u64,
) -> Result<u64> {
    let mut max_step = current_step;
    for series in response.examples {
        if series.records.is_empty() {
            continue;
        }
        let path = example_path(run_dir, &series.name);
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

fn write_system_points(run_dir: &Path, response: SystemResponse, current_step: u64) -> Result<u64> {
    let mut max_step = current_step;
    if response.points.is_empty() {
        return Ok(max_step);
    }
    let path = run_dir.join("system.csv");
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

fn write_meta(run_dir: &Path, run: &RunInfo, remote_url: &str) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Meta<'a> {
        project: &'a str,
        run_name: &'a str,
        config: serde_json::Value,
        started_at: &'a str,
        finished_at: Option<String>,
        status: &'a str,
        remote_url: &'a str,
    }

    let meta = Meta {
        project: &run.project,
        run_name: &run.name,
        config: run.config.clone(),
        started_at: &run.started_at,
        finished_at: run.finished_at.clone(),
        status: &run.status,
        remote_url,
    };
    let path = run_dir.join("meta.json");
    let content = serde_json::to_vec_pretty(&meta)?;
    fs::write(path, content)?;
    Ok(())
}

fn update_meta(run_dir: &Path, run: &RunInfo) -> Result<()> {
    let path = run_dir.join("meta.json");
    if !path.exists() {
        return Ok(());
    }
    let content = fs::read_to_string(&path)?;
    let mut meta: serde_json::Value = serde_json::from_str(&content)?;
    if let Some(obj) = meta.as_object_mut() {
        obj.insert(
            "finished_at".to_string(),
            serde_json::json!(run.finished_at),
        );
        obj.insert("status".to_string(), serde_json::json!(run.status));
    }
    let updated = serde_json::to_vec_pretty(&meta)?;
    fs::write(path, updated)?;
    Ok(())
}

fn metric_path(run_dir: &Path, metric: &str) -> PathBuf {
    let safe = sanitize_metric_name(metric);
    run_dir.join("metrics").join(format!("{}.csv", safe))
}

fn example_path(run_dir: &Path, group: &str) -> PathBuf {
    let safe = sanitize_metric_name(group);
    run_dir.join("examples").join(format!("{}.jsonl", safe))
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

fn read_cursor_from_disk(run_dir: &Path) -> RunCursor {
    RunCursor {
        metric_step: read_max_step_from_metrics(run_dir),
        example_step: read_max_step_from_examples(run_dir),
        system_step: read_row_count(&run_dir.join("system.csv")),
    }
}

fn read_max_step_from_metrics(run_dir: &Path) -> u64 {
    let metrics_dir = run_dir.join("metrics");
    if !metrics_dir.exists() {
        return 0;
    }
    find_csv_files(&metrics_dir)
        .into_iter()
        .filter_map(|path| read_max_step_from_csv(&path))
        .max()
        .unwrap_or(0)
}

fn read_max_step_from_examples(run_dir: &Path) -> u64 {
    let examples_dir = run_dir.join("examples");
    if !examples_dir.exists() {
        return 0;
    }
    find_jsonl_files(&examples_dir)
        .into_iter()
        .filter_map(|path| read_max_step_from_jsonl(&path))
        .max()
        .unwrap_or(0)
}

fn find_csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_csv_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "csv") {
                files.push(path);
            }
        }
    }
    files
}

fn find_jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_jsonl_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path);
            }
        }
    }
    files
}

fn read_max_step_from_csv(path: &Path) -> Option<u64> {
    let content = fs::read_to_string(path).ok()?;
    content
        .lines()
        .skip(1) // skip header
        .filter_map(|line| line.split(',').next()?.parse::<u64>().ok())
        .max()
}

fn read_max_step_from_jsonl(path: &Path) -> Option<u64> {
    let content = fs::read_to_string(path).ok()?;
    content
        .lines()
        .filter_map(|line| {
            let obj: serde_json::Value = serde_json::from_str(line).ok()?;
            obj.get("step")?.as_u64()
        })
        .max()
}

fn read_row_count(path: &Path) -> u64 {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    let count = content.lines().skip(1).count(); // skip header
    count as u64
}
