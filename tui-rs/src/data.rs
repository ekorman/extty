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
}

// An evaluation example (simple prompt/response pair)
#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationExample {
    pub prompt: String,
    pub response: String,
}

// An evaluation snapshot
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub name: String,
    pub run_name: String,
    pub config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Map<String, serde_json::Value>>,
    pub examples: Vec<EvaluationExample>,
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
    pub examples: HashMap<String, Vec<Example>>,
    pub start_time: Option<DateTime<Local>>,
    pub end_time: Option<DateTime<Local>>,
    pub status: RunStatus,
    pub config: Option<serde_json::Value>,
    pub remote_url: Option<String>,
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
    remote_url: Option<String>,
}

// Get the directory where local runs are stored
fn runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ex")
        .join("runs")
}

// Get the directory where remote runs are cached
fn remote_runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ex")
        .join("remote_runs")
}

// Load all runs from both local and remote directories
pub fn load_runs() -> Vec<Run> {
    let mut runs = Vec::new();

    // Load from local runs directory
    runs.extend(load_runs_from_dir(&runs_dir()));

    // Load from remote runs directory
    runs.extend(load_runs_from_dir(&remote_runs_dir()));

    // Sort by name (which includes timestamp) descending
    runs.sort_by(|a, b| b.name.cmp(&a.name));
    runs
}

// Load all runs from a specific directory (supports both flat and nested structures)
fn load_runs_from_dir(dir: &Path) -> Vec<Run> {
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

            // Check if this is a run directory (has meta.json) or project directory
            if path.join("meta.json").exists() {
                // Old flat structure: runs/<run_name>/
                if let Some(run) = load_run(&path) {
                    runs.push(run);
                }
            } else {
                // New nested structure: runs/<project>/<run_name>/
                if let Ok(run_entries) = fs::read_dir(&path) {
                    for run_entry in run_entries.flatten() {
                        let run_path = run_entry.path();
                        if run_path.is_dir()
                            && let Some(run) = load_run(&run_path)
                        {
                            runs.push(run);
                        }
                    }
                }
            }
        }
    }

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

    let (project, start_time, end_time, status, config, remote_url) = load_run_meta(path);

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
        remote_url,
    })
}

// Load all evaluations from a run directory
pub fn load_evaluations_for_run(run_path: &Path) -> Vec<Evaluation> {
    let evaluations_dir = run_path.join("evaluations");
    if !evaluations_dir.exists() {
        return vec![];
    }

    let run_name = run_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut evaluations = Vec::new();
    if let Ok(entries) = fs::read_dir(&evaluations_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Some(eval) = load_evaluation(&path, &run_name)
            {
                evaluations.push(eval);
            }
        }
    }
    evaluations.sort_by(|a, b| a.name.cmp(&b.name));
    evaluations
}

// Load a single evaluation from a JSON file
fn load_evaluation(path: &Path, run_name: &str) -> Option<Evaluation> {
    let name = path.file_stem()?.to_string_lossy().to_string();
    let content = fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&content).ok()?;

    let config = data.get("config").cloned();
    let metrics = data.get("metrics").and_then(|v| v.as_object()).cloned();
    let examples: Vec<EvaluationExample> = data
        .get("examples")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    Some(Evaluation {
        name,
        run_name: run_name.to_string(),
        config,
        metrics,
        examples,
    })
}

// Load all evaluations from both local and remote directories
pub fn load_all_evaluations() -> Vec<Evaluation> {
    let mut evaluations = Vec::new();

    // Load from local runs directory
    evaluations.extend(load_evaluations_from_dir(&runs_dir()));

    // Load from remote runs directory
    evaluations.extend(load_evaluations_from_dir(&remote_runs_dir()));

    evaluations
}

// Load all evaluations from all runs in a specific directory
fn load_evaluations_from_dir(dir: &Path) -> Vec<Evaluation> {
    if !dir.exists() {
        return vec![];
    }

    let mut evaluations = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if path.join("meta.json").exists() {
                // Old flat structure: runs/<run_name>/
                evaluations.extend(load_evaluations_for_run(&path));
            } else {
                // New nested structure: runs/<project>/<run_name>/
                if let Ok(run_entries) = fs::read_dir(&path) {
                    for run_entry in run_entries.flatten() {
                        let run_path = run_entry.path();
                        if run_path.is_dir() {
                            evaluations.extend(load_evaluations_for_run(&run_path));
                        }
                    }
                }
            }
        }
    }

    evaluations
}

/// Return type for run metadata
type RunMetadata = (
    Option<String>,            // project
    Option<DateTime<Local>>,   // start_time
    Option<DateTime<Local>>,   // end_time
    RunStatus,                 // status
    Option<serde_json::Value>, // config
    Option<String>,            // remote_url
);

fn load_run_meta(path: &Path) -> RunMetadata {
    let meta_path = path.join("meta.json");
    let Ok(content) = fs::read_to_string(&meta_path) else {
        return (None, None, None, RunStatus::Unknown, None, None);
    };

    let Ok(meta) = serde_json::from_str::<RunMeta>(&content) else {
        return (None, None, None, RunStatus::Unknown, None, None);
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

    (
        meta.project,
        start_time,
        end_time,
        status,
        meta.config,
        meta.remote_url,
    )
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
                prompts: row.data.prompt,
                responses: row.data.response,
                rewards: row.data.reward,
            });
        }
    }

    Ok(examples)
}
