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

// An evaluation example (supports batched prompts and grouped responses like training examples)
#[derive(Debug, Clone)]
pub struct EvaluationExample {
    pub prompts: Vec<String>,
    pub responses: Vec<Vec<String>>,
    pub rewards: Option<Vec<Vec<Reward>>>,
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

// A checkpoint saved during training
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub step: u64,
    pub timestamp: Option<DateTime<Local>>,
    pub size_bytes: Option<u64>,
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
    pub checkpoints: Vec<Checkpoint>,
    pub data_loaded: bool,
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
    })
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
    let metrics = load_metrics(path);
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

    let mut checkpoints: Vec<Checkpoint> = entries
        .iter()
        .filter_map(|entry| {
            let step = entry.get("step")?.as_u64()?;
            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(parse_datetime);
            let size_bytes = entry.get("size_bytes").and_then(|v| v.as_u64());
            Some(Checkpoint {
                step,
                timestamp,
                size_bytes,
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
