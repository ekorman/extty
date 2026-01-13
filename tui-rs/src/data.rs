use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};
use serde::Deserialize;

// A single data point in a metric time series
#[derive(Debug, Clone)]
pub struct MetricPoint {
    pub step: u64,
    pub value: f64,
}

// A prompt/response example
#[derive(Debug, Clone)]
pub struct Example {
    pub step: u64,
    pub prompt: String,
    pub responses: Vec<String>,
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
    #[allow(dead_code)]
    pub path: PathBuf,
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    pub examples: HashMap<String, Vec<Example>>,
    pub start_time: Option<DateTime<Local>>,
    pub end_time: Option<DateTime<Local>>,
    pub status: RunStatus,
    pub config: Option<serde_json::Value>,
}

impl Run {
    pub fn is_running(&self) -> bool {
        self.status == RunStatus::Running
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

// Get the directory where runs are stored
fn runs_dir() -> PathBuf {
    // Default to ~/.ex/runs, can be overridden with EX_RUNS_DIR
    if let Ok(dir) = std::env::var("EX_RUNS_DIR") {
        PathBuf::from(dir)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ex")
            .join("runs")
    }
}

// Load all runs from the runs directory
pub fn load_runs() -> Vec<Run> {
    let dir = runs_dir();

    if !dir.exists() {
        return vec![];
    }

    let mut runs = Vec::new();

    // Each subdirectory is a run
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && let Some(run) = load_run(&path)
            {
                runs.push(run);
            }
        }
    }

    // Sort by name (which includes timestamp) descending
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    runs
}

// Reload a single run (public for refreshing)
pub fn reload_run(path: &Path) -> Option<Run> {
    load_run(path)
}

// Delete a run by removing its directory
// WARNING: This operation cannot be undone and will recursively delete
// all files and subdirectories within the run directory
pub fn delete_run(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_dir_all(path)
}

// Load a single run from a directory
fn load_run(path: &Path) -> Option<Run> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let metrics = load_metrics(path);
    let examples = load_examples(path);

    let (start_time, end_time, status, config) = load_run_meta(path);

    Some(Run {
        name,
        path: path.to_path_buf(),
        metrics,
        examples,
        start_time,
        end_time,
        status,
        config,
    })
}

/// Return type for run metadata
type RunMetadata = (
    Option<DateTime<Local>>,
    Option<DateTime<Local>>,
    RunStatus,
    Option<serde_json::Value>,
);

fn load_run_meta(path: &Path) -> RunMetadata {
    let meta_path = path.join("meta.json");
    let Ok(content) = fs::read_to_string(&meta_path) else {
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

// Load all examples from a run directory (recursively)
fn load_examples(run_path: &Path) -> HashMap<String, Vec<Example>> {
    let mut examples = HashMap::new();
    let examples_dir = run_path.join("examples");

    if !examples_dir.exists() {
        return examples;
    }

    load_examples_recursive(&examples_dir, &examples_dir, &mut examples);

    // Sort each group by step
    for group in examples.values_mut() {
        group.sort_by_key(|e| e.step);
    }

    examples
}

// Recursively find and load JSONL files
fn load_examples_recursive(
    base_dir: &PathBuf,
    current_dir: &PathBuf,
    examples: &mut HashMap<String, Vec<Example>>,
) {
    let Ok(entries) = fs::read_dir(current_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            load_examples_recursive(base_dir, &path, examples);
        } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            // Build name from relative path (e.g., "val/example" -> "val")
            if let Ok(relative) = path.strip_prefix(base_dir) {
                let name = relative
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| relative.with_extension("").to_string_lossy().to_string());
                let name = if name.is_empty() {
                    relative.with_extension("").to_string_lossy().to_string()
                } else {
                    name
                };
                if let Ok(file_examples) = load_examples_jsonl(&path) {
                    examples.entry(name).or_default().extend(file_examples);
                }
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

#[derive(Debug, Deserialize)]
struct ExampleData {
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    prompt: Vec<String>,
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    response: Vec<String>,
}

// Load examples from a JSONL file
fn load_examples_jsonl(path: &PathBuf) -> Result<Vec<Example>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(row) = serde_json::from_str::<ExampleRow>(&line) {
            // Join multiple prompts if present (rare case)
            let prompt = row.data.prompt.join("\n");
            examples.push(Example {
                step: row.step,
                prompt,
                responses: row.data.response,
            });
        }
    }

    Ok(examples)
}
