use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Local};
use serde::Deserialize;

// A single data point in a metric time series
#[derive(Debug, Clone)]
pub struct MetricPoint {
    pub step: u64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub enum Reward {
    Scalar(f64),
    Components(HashMap<String, f64>),
}

impl Reward {
    pub fn total(&self) -> f64 {
        match self {
            Reward::Scalar(v) => *v,
            Reward::Components(map) => map.values().sum(),
        }
    }
}

// A prompt/response example (supports batched prompts and grouped responses)
#[derive(Debug, Clone)]
pub struct Example {
    pub step: u64,
    pub prompts: Vec<String>,
    pub responses: Vec<Vec<String>>,
    pub rewards: Option<Vec<Vec<Reward>>>,
    pub groundtruth: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ExampleMeta {
    pub step: u64,
    file_idx: usize,
    byte_offset: u64,
}

#[derive(Debug)]
pub struct ExampleGroup {
    files: Vec<PathBuf>,
    pub entries: Vec<ExampleMeta>,
    avg_reward: Option<f64>,
    last: Option<Example>,
}

impl ExampleGroup {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn avg_reward(&self) -> Option<f64> {
        self.avg_reward
    }

    pub fn last(&self) -> Option<&Example> {
        self.last.as_ref()
    }

    pub fn steps(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().map(|e| e.step)
    }

    pub fn find_step(&self, step: u64) -> Option<usize> {
        self.entries.binary_search_by_key(&step, |e| e.step).ok()
    }

    pub fn load(&self, idx: usize) -> Option<Example> {
        let entry = self.entries.get(idx)?;
        let path = self.files.get(entry.file_idx)?;
        let mut file = File::open(path).ok()?;
        file.seek(SeekFrom::Start(entry.byte_offset)).ok()?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let row: ExampleRow = serde_json::from_str(&line).ok()?;
        Some(Example {
            step: row.step,
            prompts: row.data.prompt,
            responses: row.data.response,
            rewards: row.data.reward,
            groundtruth: row.data.groundtruth,
        })
    }
}

// An evaluation example (supports batched prompts and grouped responses like training examples)
#[derive(Debug, Clone)]
pub struct EvaluationExample {
    pub prompts: Vec<String>,
    pub responses: Vec<Vec<String>>,
    pub rewards: Option<Vec<Vec<Reward>>>,
    pub groundtruth: Option<Vec<String>>,
}

// An evaluation snapshot (now associated with models instead of runs)
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub name: String,
    pub model_name: String,
    pub project: String,
    pub path: PathBuf,
    pub config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Map<String, serde_json::Value>>,
    pub examples: Vec<EvaluationExample>,
    pub logged_at: Option<DateTime<Local>>,
    pub started_at: Option<DateTime<Local>>,
    pub finished_at: Option<DateTime<Local>>,
}

// A model with its evaluations
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub project: String,
    pub path: PathBuf,
    pub config: Option<serde_json::Value>,
    #[allow(dead_code)]
    pub created_at: Option<DateTime<Local>>,
    #[allow(dead_code)]
    pub updated_at: Option<DateTime<Local>>,
}

#[derive(Debug, Clone)]
pub struct CheckpointFile {
    pub name: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub step: u64,
    pub timestamp: Option<DateTime<Local>>,
    pub files: Vec<CheckpointFile>,
    pub downloaded_files: Vec<String>,
}

impl Checkpoint {
    pub fn total_size_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        let mut has_any = false;
        for f in &self.files {
            if let Some(sz) = f.size_bytes {
                total += sz;
                has_any = true;
            }
        }
        if has_any { Some(total) } else { None }
    }

    pub fn all_downloaded(&self) -> bool {
        !self.files.is_empty()
            && self
                .files
                .iter()
                .all(|f| self.downloaded_files.contains(&f.name))
    }

    pub fn is_legacy(&self) -> bool {
        self.files.len() == 1 && self.files[0].name == "checkpoint.pt"
    }
}

// Run status
#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Running,
    Completed,
    Unknown,
}

// A training run with its metrics and examples
#[derive(Debug)]
pub struct Run {
    pub name: String,
    pub project: Option<String>,
    #[allow(dead_code)]
    pub path: PathBuf,
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    pub examples: HashMap<String, ExampleGroup>,
    pub start_time: Option<DateTime<Local>>,
    pub end_time: Option<DateTime<Local>>,
    pub status: RunStatus,
    pub config: Option<serde_json::Value>,
    pub checkpoints: Vec<Checkpoint>,
    pub data_loaded: bool,
    pub data_loaded_at: Option<Instant>,
}

impl Run {
    pub fn is_running(&self) -> bool {
        self.status == RunStatus::Running
    }

    pub fn display_name(&self) -> String {
        match &self.project {
            Some(project) => format!("{}/{}", project, self.name),
            None => self.name.clone(),
        }
    }
}

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

// Get the directory where local runs are stored
fn runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("runs")
}

// Get the directory where models are stored
fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("models")
}

// Get the directory where artifact metadata is cached
pub fn artifacts_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("artifacts")
}

// Load all runs with only metadata (no metrics/examples/checkpoints)
pub fn load_runs_lightweight() -> Vec<Run> {
    let mut runs = load_runs_from_dir_lightweight(&runs_dir());
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    runs
}

fn load_runs_from_dir_lightweight(dir: &Path) -> Vec<Run> {
    if !dir.exists() {
        return vec![];
    }

    let mut runs = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if path.join("meta.json").exists() {
                if let Some(run) = load_run_lightweight(&path) {
                    runs.push(run);
                }
            } else if let Ok(run_entries) = fs::read_dir(&path) {
                for run_entry in run_entries.flatten() {
                    let run_path = run_entry.path();
                    if run_path.is_dir()
                        && let Some(run) = load_run_lightweight(&run_path)
                    {
                        runs.push(run);
                    }
                }
            }
        }
    }

    runs
}

fn load_run_lightweight(path: &Path) -> Option<Run> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let (project, start_time, end_time, status, config) = load_run_meta(path);
    Some(Run {
        name,
        project,
        path: path.to_path_buf(),
        metrics: HashMap::new(),
        examples: HashMap::new(),
        start_time,
        end_time,
        status,
        config,
        checkpoints: vec![],
        data_loaded: false,
        data_loaded_at: None,
    })
}

// Reload a single run (public for refreshing)
pub fn reload_run(path: &Path) -> Option<Run> {
    load_run(path)
}

pub fn mark_run_completed(path: &Path) -> Result<(), std::io::Error> {
    let meta_path = path.join("meta.json");
    let mut data: serde_json::Value = if meta_path.exists() {
        let content = fs::read_to_string(&meta_path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    let now = chrono::Local::now().to_rfc3339();
    data["status"] = serde_json::json!("completed");
    data["finished_at"] = serde_json::json!(now);
    let json = serde_json::to_string_pretty(&data).map_err(std::io::Error::other)?;
    fs::write(&meta_path, json)
}

// Delete a run by removing its directory
// WARNING: This operation cannot be undone and will recursively delete
// all files and subdirectories within the run directory
pub fn delete_run(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_dir_all(path)
}

pub fn move_run(old_path: &Path, new_project: &str) -> Result<PathBuf, std::io::Error> {
    let run_name = old_path
        .file_name()
        .ok_or_else(|| std::io::Error::other("invalid run path"))?;

    let dest = if new_project.is_empty() {
        runs_dir().join(run_name)
    } else {
        runs_dir().join(new_project).join(run_name)
    };

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::rename(old_path, &dest)?;

    let meta_path = dest.join("meta.json");
    let mut data: serde_json::Value = if meta_path.exists() {
        let content = fs::read_to_string(&meta_path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    if new_project.is_empty() {
        data.as_object_mut().map(|m| m.remove("project"));
    } else {
        data["project"] = serde_json::json!(new_project);
    }

    let json = serde_json::to_string_pretty(&data).map_err(std::io::Error::other)?;
    fs::write(&meta_path, json)?;

    Ok(dest)
}

// Delete an evaluation by removing its JSON file
pub fn delete_evaluation(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_file(path)
}

// Delete a model by removing its directory
pub fn delete_model(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_dir_all(path)
}

// Load a single run from a directory
fn load_run(path: &Path) -> Option<Run> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let mut metrics = load_metrics(path);
    metrics.extend(load_system_metrics(path));
    let examples = load_examples(path);
    let checkpoints = load_checkpoints(path);

    let (project, start_time, end_time, status, config) = load_run_meta(path);

    Some(Run {
        name,
        project,
        path: path.to_path_buf(),
        metrics,
        examples,
        start_time,
        end_time,
        status,
        config,
        checkpoints,
        data_loaded: true,
        data_loaded_at: Some(Instant::now()),
    })
}

// Load all models from the models directory
pub fn load_models() -> Vec<Model> {
    let mut models = Vec::new();
    let dir = models_dir();

    if !dir.exists() {
        return models;
    }

    if let Ok(project_entries) = fs::read_dir(&dir) {
        for project_entry in project_entries.flatten() {
            let project_path = project_entry.path();
            if !project_path.is_dir() {
                continue;
            }

            let project_name = project_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            if let Ok(model_entries) = fs::read_dir(&project_path) {
                for model_entry in model_entries.flatten() {
                    let model_path = model_entry.path();
                    if model_path.is_dir()
                        && let Some(model) = load_model(&model_path, &project_name)
                    {
                        models.push(model);
                    }
                }
            }
        }
    }

    models.sort_by(|a, b| a.project.cmp(&b.project).then_with(|| b.name.cmp(&a.name)));
    models
}

// Load a single model from a directory
fn load_model(path: &Path, project: &str) -> Option<Model> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let meta_path = path.join("meta.json");

    let (config, created_at, updated_at) = if meta_path.exists() {
        let content = fs::read_to_string(&meta_path).ok()?;
        let data: serde_json::Value = serde_json::from_str(&content).ok()?;

        let config = data.get("model_config").cloned();
        let created_at = data
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Local));
        let updated_at = data
            .get("updated_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Local));

        (config, created_at, updated_at)
    } else {
        (None, None, None)
    };

    Some(Model {
        name,
        project: project.to_string(),
        path: path.to_path_buf(),
        config,
        created_at,
        updated_at,
    })
}

// Load all evaluations from a model directory
pub fn load_evaluations_for_model(
    model_path: &Path,
    model_name: &str,
    project: &str,
) -> Vec<Evaluation> {
    let evaluations_dir = model_path.join("evaluations");
    if !evaluations_dir.exists() {
        return vec![];
    }

    let mut evaluations = Vec::new();
    if let Ok(entries) = fs::read_dir(&evaluations_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(eval) = load_evaluation(&path, model_name, project)
            {
                evaluations.push(eval);
            }
        }
    }
    evaluations.sort_by(|a, b| a.name.cmp(&b.name));
    evaluations
}

// Load a single evaluation from a JSON file
fn load_evaluation(path: &Path, model_name: &str, project: &str) -> Option<Evaluation> {
    let name = path.file_stem()?.to_string_lossy().to_string();
    let content = fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&content).ok()?;

    let config = data.get("config").cloned();
    let metrics = data.get("metrics").and_then(|v| v.as_object()).cloned();
    let examples: Vec<EvaluationExample> = data
        .get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|ex| {
                    let ex_data: ExampleData = serde_json::from_value(ex.clone()).ok()?;
                    Some(EvaluationExample {
                        prompts: ex_data.prompt,
                        responses: ex_data.response,
                        rewards: ex_data.reward,
                        groundtruth: ex_data.groundtruth,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let logged_at = data
        .get("logged_at")
        .and_then(|v| v.as_str())
        .and_then(parse_datetime);
    let started_at = data
        .get("started_at")
        .and_then(|v| v.as_str())
        .and_then(parse_datetime);
    let finished_at = data
        .get("finished_at")
        .and_then(|v| v.as_str())
        .and_then(parse_datetime);

    Some(Evaluation {
        name,
        model_name: model_name.to_string(),
        project: project.to_string(),
        path: path.to_path_buf(),
        config,
        metrics,
        examples,
        logged_at,
        started_at,
        finished_at,
    })
}

/// Parse a datetime string in either RFC3339 or ISO format
fn parse_datetime(s: &str) -> Option<DateTime<Local>> {
    // Try RFC3339 first (with timezone)
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Local));
    }
    // Try ISO format without timezone (assume local)
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return dt.and_local_timezone(Local).single();
    }
    // Try without fractional seconds
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return dt.and_local_timezone(Local).single();
    }
    None
}

// Load all evaluations from all models
pub fn load_all_evaluations() -> Vec<Evaluation> {
    let mut evaluations = Vec::new();

    let dir = models_dir();
    if !dir.exists() {
        return evaluations;
    }

    if let Ok(project_entries) = fs::read_dir(&dir) {
        for project_entry in project_entries.flatten() {
            let project_path = project_entry.path();
            if !project_path.is_dir() {
                continue;
            }

            let project_name = project_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            if let Ok(model_entries) = fs::read_dir(&project_path) {
                for model_entry in model_entries.flatten() {
                    let model_path = model_entry.path();
                    if model_path.is_dir() {
                        let model_name = model_path
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        evaluations.extend(load_evaluations_for_model(
                            &model_path,
                            &model_name,
                            &project_name,
                        ));
                    }
                }
            }
        }
    }

    evaluations
}

// Load checkpoints from checkpoints.json
fn load_checkpoints(run_path: &Path) -> Vec<Checkpoint> {
    let checkpoints_path = run_path.join("checkpoints.json");
    if !checkpoints_path.exists() {
        return vec![];
    }

    let Ok(content) = fs::read_to_string(&checkpoints_path) else {
        return vec![];
    };

    let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(&content) else {
        return vec![];
    };

    let checkpoints_dir = run_path.join("checkpoints");
    let mut checkpoints: Vec<Checkpoint> = entries
        .iter()
        .filter_map(|entry| {
            let step = entry.get("step")?.as_u64()?;
            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(parse_datetime);

            let files: Vec<CheckpointFile> = match entry.get("files") {
                Some(serde_json::Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|item| match item {
                        serde_json::Value::String(name) => Some(CheckpointFile {
                            name: name.clone(),
                            size_bytes: None,
                        }),
                        serde_json::Value::Object(obj) => {
                            let name = obj.get("name")?.as_str()?.to_string();
                            let size_bytes = obj.get("size_bytes").and_then(|v| v.as_u64());
                            Some(CheckpointFile { name, size_bytes })
                        }
                        _ => None,
                    })
                    .collect(),
                _ => {
                    let size_bytes = entry.get("size_bytes").and_then(|v| v.as_u64());
                    vec![CheckpointFile {
                        name: "checkpoint.pt".to_string(),
                        size_bytes,
                    }]
                }
            };

            let step_dir = checkpoints_dir.join(step.to_string());
            let downloaded_files: Vec<String> = files
                .iter()
                .filter(|f| step_dir.join(&f.name).exists())
                .map(|f| f.name.clone())
                .collect();

            Some(Checkpoint {
                step,
                timestamp,
                files,
                downloaded_files,
            })
        })
        .collect();

    checkpoints.sort_by(|a, b| b.step.cmp(&a.step));
    checkpoints
}

/// Return type for run metadata
type RunMetadata = (
    Option<String>,            // project
    Option<DateTime<Local>>,   // start_time
    Option<DateTime<Local>>,   // end_time
    RunStatus,                 // status
    Option<serde_json::Value>, // config
);

fn load_run_meta(path: &Path) -> RunMetadata {
    let meta_path = path.join("meta.json");
    let Ok(content) = fs::read_to_string(&meta_path) else {
        return (None, None, None, RunStatus::Unknown, None);
    };

    let Ok(meta) = serde_json::from_str::<RunMeta>(&content) else {
        return (None, None, None, RunStatus::Unknown, None);
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

    (meta.project, start_time, end_time, status, meta.config)
}

// Load all metrics from a run directory (recursively)
fn load_metrics(run_path: &Path) -> HashMap<String, Vec<MetricPoint>> {
    let mut metrics = HashMap::new();
    let metrics_dir = run_path.join("metrics");

    if !metrics_dir.exists() {
        return metrics;
    }

    load_metrics_recursive(&metrics_dir, &metrics_dir, &mut metrics);
    metrics
}

// Recursively find and load CSV files
fn load_metrics_recursive(
    base_dir: &PathBuf,
    current_dir: &PathBuf,
    metrics: &mut HashMap<String, Vec<MetricPoint>>,
) {
    let Ok(entries) = fs::read_dir(current_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            load_metrics_recursive(base_dir, &path, metrics);
        } else if path.extension().map(|e| e == "csv").unwrap_or(false) {
            // Build metric name from relative path (e.g., "train/loss")
            if let Ok(relative) = path.strip_prefix(base_dir) {
                let name = relative.with_extension("").to_string_lossy().to_string();
                if let Ok(points) = load_metric_csv(&path) {
                    metrics.insert(name, points);
                }
            }
        }
    }
}

// CSV row format for metrics
#[derive(Debug, Deserialize)]
struct MetricRow {
    step: u64,
    #[allow(dead_code)]
    timestamp: f64,
    value: f64,
}

// Load metric data from a CSV file
fn load_metric_csv(path: &PathBuf) -> Result<Vec<MetricPoint>, csv::Error> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut points = Vec::new();

    for result in reader.deserialize() {
        let row: MetricRow = result?;
        points.push(MetricPoint {
            step: row.step,
            value: row.value,
        });
    }

    Ok(points)
}

#[derive(Debug, Deserialize)]
struct SystemMetricRow {
    timestamp: f64,
    ram_used_gb: f64,
    ram_total_gb: f64,
    gpu_mem_used_gb: f64,
    gpu_mem_total_gb: f64,
    gpu_util_pct: f64,
}

fn load_system_metrics(run_path: &Path) -> HashMap<String, Vec<MetricPoint>> {
    let mut metrics = HashMap::new();
    let path = run_path.join("system.csv");

    if !path.exists() {
        return metrics;
    }

    let Ok(mut reader) = csv::Reader::from_path(&path) else {
        return metrics;
    };

    let mut ram_used = Vec::new();
    let mut gpu_mem_used = Vec::new();
    let mut gpu_util = Vec::new();

    let mut first_ts: Option<f64> = None;

    for result in reader.deserialize() {
        let Ok(row): Result<SystemMetricRow, _> = result else {
            continue;
        };

        let ts = *first_ts.get_or_insert(row.timestamp);
        let elapsed_min = ((row.timestamp - ts) / 60.0).round() as u64;

        let ram_pct = if row.ram_total_gb > 0.0 {
            row.ram_used_gb / row.ram_total_gb * 100.0
        } else {
            0.0
        };
        ram_used.push(MetricPoint {
            step: elapsed_min,
            value: ram_pct,
        });

        let gpu_mem_pct = if row.gpu_mem_total_gb > 0.0 {
            row.gpu_mem_used_gb / row.gpu_mem_total_gb * 100.0
        } else {
            0.0
        };
        gpu_mem_used.push(MetricPoint {
            step: elapsed_min,
            value: gpu_mem_pct,
        });

        gpu_util.push(MetricPoint {
            step: elapsed_min,
            value: row.gpu_util_pct,
        });
    }

    if !ram_used.is_empty() {
        metrics.insert("sys/ram %".to_string(), ram_used);
    }
    if !gpu_mem_used.is_empty() {
        metrics.insert("sys/gpu mem %".to_string(), gpu_mem_used);
    }
    if !gpu_util.is_empty() {
        metrics.insert("sys/gpu util %".to_string(), gpu_util);
    }

    metrics
}

type ExampleGroupBuilder = (Vec<PathBuf>, Vec<ExampleMeta>, f64, usize, Option<Example>);

fn load_examples(run_path: &Path) -> HashMap<String, ExampleGroup> {
    let mut groups: HashMap<String, ExampleGroupBuilder> = HashMap::new();
    let examples_dir = run_path.join("examples");

    if !examples_dir.exists() {
        return HashMap::new();
    }

    load_examples_recursive(&examples_dir, &examples_dir, &mut groups);

    groups
        .into_iter()
        .map(
            |(name, (files, mut entries, reward_total, reward_count, last))| {
                entries.sort_by_key(|e| e.step);
                let avg_reward = if reward_count > 0 {
                    Some(reward_total / reward_count as f64)
                } else {
                    None
                };
                let last = if let Some(max_entry) = entries.last() {
                    if last.as_ref().map(|l| l.step) == Some(max_entry.step) {
                        last
                    } else {
                        let group = ExampleGroup {
                            files: files.clone(),
                            entries: vec![max_entry.clone()],
                            avg_reward: None,
                            last: None,
                        };
                        group.load(0)
                    }
                } else {
                    None
                };
                (
                    name,
                    ExampleGroup {
                        files,
                        entries,
                        avg_reward,
                        last,
                    },
                )
            },
        )
        .collect()
}

fn load_examples_recursive(
    base_dir: &PathBuf,
    current_dir: &PathBuf,
    groups: &mut HashMap<String, ExampleGroupBuilder>,
) {
    let Ok(entries) = fs::read_dir(current_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            load_examples_recursive(base_dir, &path, groups);
        } else if path.extension().map(|e| e == "jsonl").unwrap_or(false)
            && let Ok(relative) = path.strip_prefix(base_dir)
        {
            let name = relative
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| relative.with_extension("").to_string_lossy().to_string());
            let name = if name.is_empty() {
                relative.with_extension("").to_string_lossy().to_string()
            } else {
                name
            };
            let group = groups
                .entry(name)
                .or_insert_with(|| (vec![], vec![], 0.0, 0, None));
            let file_idx = group.0.len();
            group.0.push(path.clone());
            if let Ok((metas, reward_total, reward_count, last_example)) =
                build_example_index(&path, file_idx)
            {
                group.1.extend(metas);
                group.2 += reward_total;
                group.3 += reward_count;
                group.4 = last_example;
            }
        }
    }
}

// JSONL row format for examples
#[derive(Debug, Deserialize)]
struct ExampleRow {
    step: u64,
    #[allow(dead_code)]
    timestamp: f64,
    data: ExampleData,
}

/// Helper to deserialize a value that can be either a single string or a list of strings
fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut strings = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                strings.push(s);
            }
            Ok(strings)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

/// Helper to deserialize an optional value that can be either a single string or a list of strings
fn deserialize_optional_string_or_vec<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct OptionalStringOrVec;

    impl<'de> de::Visitor<'de> for OptionalStringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("null, a string, or array of strings")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(vec![value.to_owned()]))
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(vec![value]))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut strings = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                strings.push(s);
            }
            Ok(Some(strings))
        }
    }

    deserializer.deserialize_any(OptionalStringOrVec)
}

/// Helper to deserialize responses: string, array of strings, or array of arrays of strings
/// - "r1" → [[r1]]
/// - ["r1", "r2"] → [[r1, r2]] (old format: variants for single prompt)
/// - [["r1a", "r1b"], ["r2a"]] → as-is (new format: batch of prompts with variants)
fn deserialize_responses<'de, D>(deserializer: D) -> Result<Vec<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ResponsesVisitor;

    impl<'de> de::Visitor<'de> for ResponsesVisitor {
        type Value = Vec<Vec<String>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string, array of strings, or array of arrays of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![vec![value.to_owned()]])
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![vec![value]])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut result: Vec<Vec<String>> = Vec::new();
            let mut is_nested: Option<bool> = None;
            let mut flat_strings: Vec<String> = Vec::new();

            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                match elem {
                    serde_json::Value::String(s) => {
                        if is_nested == Some(true) {
                            return Err(de::Error::custom(
                                "mixed string and array elements in response",
                            ));
                        }
                        is_nested = Some(false);
                        flat_strings.push(s);
                    }
                    serde_json::Value::Array(arr) => {
                        if is_nested == Some(false) {
                            return Err(de::Error::custom(
                                "mixed string and array elements in response",
                            ));
                        }
                        is_nested = Some(true);
                        let inner: Vec<String> = arr
                            .into_iter()
                            .map(|v| match v {
                                serde_json::Value::String(s) => Ok(s),
                                _ => Err(de::Error::custom("expected string in inner array")),
                            })
                            .collect::<Result<_, _>>()?;
                        result.push(inner);
                    }
                    _ => {
                        return Err(de::Error::custom("expected string or array in response"));
                    }
                }
            }

            if is_nested == Some(false) || is_nested.is_none() {
                Ok(vec![flat_strings])
            } else {
                Ok(result)
            }
        }
    }

    deserializer.deserialize_any(ResponsesVisitor)
}

/// Parse a single reward value (number or object) from a serde_json::Value
fn parse_single_reward<E: serde::de::Error>(elem: serde_json::Value) -> Result<Reward, E> {
    match elem {
        serde_json::Value::Number(n) => {
            Ok(Reward::Scalar(n.as_f64().ok_or_else(|| {
                E::custom("expected f64-compatible number in reward")
            })?))
        }
        serde_json::Value::Object(obj) => {
            let mut components = HashMap::new();
            for (k, v) in obj {
                let val = v
                    .as_f64()
                    .ok_or_else(|| E::custom("expected f64 value in reward object"))?;
                components.insert(k, val);
            }
            Ok(Reward::Components(components))
        }
        _ => Err(E::custom("expected number or object for reward")),
    }
}

/// Helper to deserialize rewards: supports nested arrays for batched examples
/// - 0.5 → [[Scalar(0.5)]]
/// - {"a": 0.5} → [[Components({a: 0.5})]]
/// - [0.5, 0.6] → [[Scalar(0.5), Scalar(0.6)]] (flat array for single prompt)
/// - [[0.5, 0.6], [0.7]] → as-is (nested for batched prompts)
fn deserialize_rewards<'de, D>(deserializer: D) -> Result<Option<Vec<Vec<Reward>>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct RewardsVisitor;

    impl<'de> de::Visitor<'de> for RewardsVisitor {
        type Value = Option<Vec<Vec<Reward>>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number, object, array of numbers/objects, or nested array")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(vec![vec![Reward::Scalar(value)]]))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(vec![vec![Reward::Scalar(value as f64)]]))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(vec![vec![Reward::Scalar(value as f64)]]))
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let mut components = HashMap::new();
            while let Some((key, value)) = map.next_entry::<String, f64>()? {
                components.insert(key, value);
            }
            Ok(Some(vec![vec![Reward::Components(components)]]))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut outer: Vec<Vec<Reward>> = Vec::new();
            let mut is_nested: Option<bool> = None;
            let mut flat_rewards: Vec<Reward> = Vec::new();

            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                match &elem {
                    serde_json::Value::Array(inner_arr) => {
                        if is_nested == Some(false) {
                            return Err(de::Error::custom(
                                "mixed nested and flat elements in reward array",
                            ));
                        }
                        is_nested = Some(true);
                        let mut inner_rewards = Vec::new();
                        for inner_elem in inner_arr.clone() {
                            inner_rewards.push(parse_single_reward(inner_elem)?);
                        }
                        outer.push(inner_rewards);
                    }
                    _ => {
                        if is_nested == Some(true) {
                            return Err(de::Error::custom(
                                "mixed nested and flat elements in reward array",
                            ));
                        }
                        is_nested = Some(false);
                        flat_rewards.push(parse_single_reward(elem)?);
                    }
                }
            }

            if flat_rewards.is_empty() && outer.is_empty() {
                Ok(None)
            } else if is_nested == Some(true) {
                Ok(Some(outer))
            } else {
                Ok(Some(vec![flat_rewards]))
            }
        }
    }

    deserializer.deserialize_any(RewardsVisitor)
}

#[derive(Debug, Deserialize)]
struct ExampleData {
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    prompt: Vec<String>,
    #[serde(deserialize_with = "deserialize_responses")]
    response: Vec<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_rewards")]
    reward: Option<Vec<Vec<Reward>>>,
    #[serde(default, deserialize_with = "deserialize_optional_string_or_vec")]
    groundtruth: Option<Vec<String>>,
}

fn build_example_index(
    path: &PathBuf,
    file_idx: usize,
) -> Result<(Vec<ExampleMeta>, f64, usize, Option<Example>), std::io::Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut metas = Vec::new();
    let mut reward_total = 0.0;
    let mut reward_count = 0;
    let mut last_example = None;
    let mut line = String::new();

    loop {
        let byte_offset = reader.stream_position()?;
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if let Ok(row) = serde_json::from_str::<ExampleRow>(&line) {
            metas.push(ExampleMeta {
                step: row.step,
                file_idx,
                byte_offset,
            });
            if let Some(ref rewards) = row.data.reward {
                for prompt_rewards in rewards {
                    for reward in prompt_rewards {
                        reward_total += reward.total();
                        reward_count += 1;
                    }
                }
            }
            last_example = Some(Example {
                step: row.step,
                prompts: row.data.prompt,
                responses: row.data.response,
                rewards: row.data.reward,
                groundtruth: row.data.groundtruth,
            });
        }
    }

    Ok((metas, reward_total, reward_count, last_example))
}

fn starred_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("starred.json")
}

pub fn load_starred_runs() -> HashSet<String> {
    let path = starred_path();
    let Ok(content) = fs::read_to_string(&path) else {
        return HashSet::new();
    };
    let Ok(names) = serde_json::from_str::<Vec<String>>(&content) else {
        return HashSet::new();
    };
    names.into_iter().collect()
}

pub fn save_starred_runs(starred: &HashSet<String>) {
    let path = starred_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let names: Vec<&String> = starred.iter().collect();
    if let Ok(json) = serde_json::to_string(&names) {
        let _ = fs::write(&path, json);
    }
}

fn notes_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("notes.json")
}

pub fn load_run_notes() -> HashMap<String, String> {
    let path = notes_path();
    let Ok(content) = fs::read_to_string(&path) else {
        return HashMap::new();
    };
    let Ok(notes) = serde_json::from_str::<HashMap<String, String>>(&content) else {
        return HashMap::new();
    };
    notes
}

pub fn save_run_notes(notes: &HashMap<String, String>) {
    let path = notes_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(notes) {
        let _ = fs::write(&path, json);
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactFile {
    pub path: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub name: String,
    pub description: String,
    pub content_type: String,
    pub tags: Vec<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub total_size_bytes: Option<u64>,
    pub files: Vec<ArtifactFile>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Artifact {
    pub fn display_size(&self) -> String {
        match self.total_size_bytes {
            Some(b) if b >= 1_073_741_824 => format!("{:.1} GB", b as f64 / 1_073_741_824.0),
            Some(b) if b >= 1_048_576 => format!("{:.1} MB", b as f64 / 1_048_576.0),
            Some(b) if b >= 1024 => format!("{:.1} KB", b as f64 / 1024.0),
            Some(b) => format!("{} B", b),
            None => "—".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ArtifactMeta {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_content_type")]
    content_type: String,
    #[serde(default)]
    tags: Vec<String>,
    created_at: Option<String>,
    updated_at: Option<String>,
    total_size_bytes: Option<u64>,
    #[serde(default)]
    files: Vec<ArtifactFileMeta>,
    #[serde(default)]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

fn default_content_type() -> String {
    "file".to_string()
}

#[derive(Debug, Deserialize)]
struct ArtifactFileMeta {
    path: String,
    size_bytes: Option<u64>,
}

pub fn load_artifacts_from_cache() -> Vec<Artifact> {
    let dir = artifacts_dir();
    if !dir.exists() {
        return vec![];
    }

    let mut artifacts = Vec::new();
    let Ok(entries) = fs::read_dir(&dir) else {
        return vec![];
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let meta_path = path.join("meta.json");
        if !meta_path.exists() {
            continue;
        }
        let Ok(content) = fs::read_to_string(&meta_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<ArtifactMeta>(&content) else {
            continue;
        };
        artifacts.push(Artifact {
            name: meta.name,
            description: meta.description,
            content_type: meta.content_type,
            tags: meta.tags,
            created_at: meta.created_at,
            updated_at: meta.updated_at,
            total_size_bytes: meta.total_size_bytes,
            files: meta
                .files
                .into_iter()
                .map(|f| ArtifactFile {
                    path: f.path,
                    size_bytes: f.size_bytes,
                })
                .collect(),
            metadata: meta.metadata,
        });
    }

    artifacts.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    artifacts
}
