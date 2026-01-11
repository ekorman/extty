use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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
    pub response: String,
}

// A training run with its metrics and examples
#[derive(Debug)]
pub struct Run {
    pub name: String,
    #[allow(dead_code)]
    pub path: PathBuf,
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    pub examples: HashMap<String, Vec<Example>>,
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
            if path.is_dir() {
                if let Some(run) = load_run(&path) {
                    runs.push(run);
                }
            }
        }
    }

    // Sort by name (which includes timestamp) descending
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    runs
}

// Load a single run from a directory
fn load_run(path: &PathBuf) -> Option<Run> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let metrics = load_metrics(path);
    let examples = load_examples(path);

    Some(Run {
        name,
        path: path.clone(),
        metrics,
        examples,
    })
}

// Load all metrics from a run directory (recursively)
fn load_metrics(run_path: &PathBuf) -> HashMap<String, Vec<MetricPoint>> {
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
    let Ok(entries) = fs::read_dir(current_dir) else { return };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            load_metrics_recursive(base_dir, &path, metrics);
        } else if path.extension().map(|e| e == "csv").unwrap_or(false) {
            // Build metric name from relative path (e.g., "train/loss")
            if let Ok(relative) = path.strip_prefix(base_dir) {
                let name = relative
                    .with_extension("")
                    .to_string_lossy()
                    .to_string();
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
fn load_examples(run_path: &PathBuf) -> HashMap<String, Vec<Example>> {
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
    let Ok(entries) = fs::read_dir(current_dir) else { return };

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
                    .unwrap_or_else(|| {
                        relative
                            .with_extension("")
                            .to_string_lossy()
                            .to_string()
                    });
                let name = if name.is_empty() {
                    relative.with_extension("").to_string_lossy().to_string()
                } else {
                    name
                };
                if let Ok(file_examples) = load_examples_jsonl(&path) {
                    examples
                        .entry(name)
                        .or_insert_with(Vec::new)
                        .extend(file_examples);
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

#[derive(Debug, Deserialize)]
struct ExampleData {
    prompt: String,
    response: String,
}

// Load examples from a JSONL file
fn load_examples_jsonl(path: &PathBuf) -> Result<Vec<Example>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut examples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(row) = serde_json::from_str::<ExampleRow>(&line) {
            examples.push(Example {
                step: row.step,
                prompt: row.data.prompt,
                response: row.data.response,
            });
        }
    }

    Ok(examples)
}
