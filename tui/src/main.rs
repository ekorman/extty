use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    prelude::*,
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph},
};

// Cyberpunk color palette
const NEON_CYAN: Color = Color::Rgb(0, 255, 255);
const NEON_MAGENTA: Color = Color::Rgb(255, 0, 128);
const NEON_GREEN: Color = Color::Rgb(0, 255, 136);
const NEON_YELLOW: Color = Color::Rgb(255, 255, 0);
const DIM_CYAN: Color = Color::Rgb(0, 139, 139);

type CompareRunData = (String, Color, Vec<(f64, f64)>, bool);

const COMPARE_COLORS: [Color; 8] = [
    NEON_GREEN,
    NEON_CYAN,
    NEON_MAGENTA,
    NEON_YELLOW,
    Color::Rgb(255, 128, 0),
    Color::Rgb(180, 0, 255),
    Color::Rgb(255, 100, 100),
    Color::Rgb(0, 200, 255),
];

mod data;
mod infra;
mod prune;
mod run;
mod s3;
use data::{
    Artifact, Checkpoint, ConfusionMatrixPoint, Example, ExampleGroup, MetricPoint, Reward, Run,
    artifacts_dir, load_archived_projects, load_artifacts_from_cache, load_run_notes,
    load_runs_lightweight, load_starred_runs, mark_run_completed, save_archived_projects,
    save_run_notes, save_starred_runs,
};
use infra::{
    InfraConfig, Instance, InstanceStatus, InstanceType, LocalMachine, Provider, ScriptOptions,
    generate_script, get_provider, load_config, project_remote_dir, save_config,
};
enum SetupMessage {
    Status(String),
    Done(String),
    Error(String),
}

enum S3PullMessage {
    Pulling(String),
    Done(String),
    Error(String),
}

// View mode: Runs, Artifacts, or Infra
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Runs,
    Artifacts,
    Infra,
}

// The views in our app
#[derive(Clone, Copy, PartialEq)]
enum View {
    List,
    RunDetail,
    Focused,
    Compare,
    ConfigFull,
    InfraList,
    InfraConfig,
    S3Config,
    ArtifactDetail,
}

// Which panel has focus in the Infra dashboard
#[derive(Clone, Copy, PartialEq)]
enum InfraPanel {
    Instances,
    Types,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum InfraTypeSort {
    #[default]
    Name,
    Price,
    Vram,
}

impl InfraTypeSort {
    fn next(self) -> Self {
        match self {
            Self::Name => Self::Price,
            Self::Price => Self::Vram,
            Self::Vram => Self::Name,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Price => "Price",
            Self::Vram => "VRAM",
        }
    }
}

// Which section is focused in the focused example view
#[derive(Clone, Copy, PartialEq)]
enum FocusedSection {
    Prompt,
    Groundtruth,
    Response,
}

// Which field is focused in the session setup modal
#[derive(Clone, Copy, PartialEq, Default)]
enum SessionModalField {
    #[default]
    PythonVersion,
    RepoPath,
    Command,
    SkipTmux,
}

// Which field is focused in the add machine modal
#[derive(Clone, Copy, PartialEq, Default)]
enum AddMachineField {
    #[default]
    User,
    Host,
    Name,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum FilterPhase {
    #[default]
    KeySelect,
    ValueSelect,
}

// A card in the detail grid - chart, example group, checkpoints, or artifacts
#[derive(Clone)]
enum Card {
    Chart { name: String },
    Examples { name: String },
    ConfusionMatrix { name: String },
    Checkpoints,
    Artifacts,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum XAxis {
    Step,
    Timestamp,
}

// Represents an item in the hierarchical list view (runs)
#[derive(Clone, Debug)]
enum ListEntry {
    Project { name: String },
    Run { run_index: usize },
}

// All application state lives here
struct App {
    runs: Vec<Run>,
    selected_run: usize,
    selected_list_item: usize, // Current position in the flattened list (runs)
    list_state: ListState,
    expanded_projects: HashSet<String>, // Which projects are expanded (runs)
    selected_card: usize,
    selected_example: usize,  // Index within an example group when focused
    selected_prompt: usize,   // Index within prompts batch for an example
    selected_response: usize, // Index within response variants for an example
    scroll_offset: usize,
    focused_section: FocusedSection, // Which section (prompt/response/groundtruth) is focused
    prompt_scroll_offset: usize,     // Scroll offset for prompt in focused view
    response_scroll_offset: usize,   // Scroll offset for response in focused view
    groundtruth_scroll_offset: usize, // Scroll offset for groundtruth in focused view
    view: View,
    view_mode: ViewMode,
    should_quit: bool,
    show_config: bool,
    config_panel_scroll: usize,
    show_delete_confirm: bool,
    show_complete_confirm: bool,
    pending_delete_run: Option<usize>, // Index into runs vector of run to delete
    term_width: u16,
    term_height: u16,
    // Infra state
    infra_config: InfraConfig,
    infra_instances: Vec<Instance>,
    infra_types: Vec<InstanceType>,
    selected_infra_provider: Provider,
    selected_infra_instance: usize,
    selected_infra_type: usize,
    infra_types_list_state: ListState,
    infra_loading: bool,
    infra_error: Option<String>,
    infra_message_time: Option<Instant>,
    show_terminate_confirm: bool,
    pending_terminate_instance: Option<usize>,
    infra_active_panel: InfraPanel,
    infra_type_sort: InfraTypeSort,
    infra_last_refresh: Instant,
    // Infra config editing
    config_provider_index: usize,
    config_api_key_input: String,
    config_editing_key: bool,
    // Infra launch (inline in types panel)
    launch_selected_region: usize,
    launch_name_input: String,
    launch_confirming: bool,
    launch_selecting_region: bool,
    // S3 config editing
    s3_config: s3::S3Config,
    s3_config_field: usize,
    s3_config_editing: bool,
    s3_config_input: String,
    s3_config_message: Option<String>,
    // Add local machine modal
    add_machine_open: bool,
    add_machine_name: String,
    add_machine_user: String,
    add_machine_host: String,
    add_machine_focus: AddMachineField,
    // Session setup modal
    session_modal_open: bool,
    session_modal_instance: Option<Instance>,
    session_python_version: String,
    session_repo_path: String,
    session_command: String,
    session_skip_tmux: bool,
    session_modal_focus: SessionModalField,
    setup_rx: Option<mpsc::Receiver<SetupMessage>>,
    compared_runs: Vec<usize>,
    starred_runs: HashSet<String>,
    run_notes: HashMap<String, String>,
    compare_focused: bool,
    s3_pull_rx: Option<mpsc::Receiver<S3PullMessage>>,
    s3_pull_status: Option<String>,
    s3_pull_time: Option<Instant>,
    cached_example: Option<(String, usize, Example)>,
    goto_step_input: Option<String>,
    pull_run_modal_open: bool,
    note_modal_open: bool,
    note_modal_input: String,
    note_modal_run_name: String,
    move_run_modal_open: bool,
    move_run_input: String,
    move_run_index: Option<usize>,
    pull_run_input: String,
    config_cursor: usize,
    config_copied_at: Option<Instant>,
    selected_checkpoint: usize,
    checkpoint_download_modal: bool,
    checkpoint_download_options: Vec<String>,
    checkpoint_download_selected: usize,
    checkpoint_download_step: u64,
    show_system_metrics: bool,
    timestamp_x_metrics: HashSet<String>,
    config_filters: Vec<(String, String)>,
    filter_modal_open: bool,
    filter_modal_phase: FilterPhase,
    filter_modal_input: String,
    filter_modal_selected: usize,
    filter_modal_key: String,
    search_query: String,
    search_editing: bool,
    hide_completed: bool,
    archived_projects: HashSet<String>,
    show_archived: bool,
    // Artifacts
    artifacts: Vec<Artifact>,
    selected_artifact: usize,
    artifact_detail_scroll: usize,
    pending_delete_artifact: bool,
    pending_delete_checkpoint: bool,
    help_overlay_open: bool,
    help_overlay_scroll: u16,
    confusion_step_idx: HashMap<String, usize>,
}

impl App {
    fn new() -> Self {
        let runs = load_runs_lightweight();
        let infra_config = load_config().unwrap_or_default();
        let default_provider = infra_config.default_provider;
        App {
            runs,
            selected_run: 0,
            selected_list_item: 0,
            list_state: ListState::default(),
            expanded_projects: HashSet::new(),
            selected_card: 0,
            selected_example: 0,
            selected_prompt: 0,
            selected_response: 0,
            scroll_offset: 0,
            focused_section: FocusedSection::Response,
            prompt_scroll_offset: 0,
            response_scroll_offset: 0,
            groundtruth_scroll_offset: 0,
            view: View::List,
            view_mode: ViewMode::Runs,
            should_quit: false,
            show_config: false,
            config_panel_scroll: 0,
            show_delete_confirm: false,
            show_complete_confirm: false,
            pending_delete_run: None,
            term_width: 80,
            term_height: 24,
            infra_config,
            infra_instances: Vec::new(),
            infra_types: Vec::new(),
            selected_infra_provider: default_provider,
            selected_infra_instance: 0,
            selected_infra_type: 0,
            infra_types_list_state: ListState::default(),
            infra_loading: false,
            infra_error: None,
            infra_message_time: None,
            show_terminate_confirm: false,
            pending_terminate_instance: None,
            infra_active_panel: InfraPanel::Instances,
            infra_type_sort: InfraTypeSort::default(),
            infra_last_refresh: Instant::now(),
            config_provider_index: 0,
            config_api_key_input: String::new(),
            config_editing_key: false,
            launch_selected_region: 0,
            launch_name_input: String::new(),
            launch_confirming: false,
            launch_selecting_region: false,
            s3_config: s3::load_config().ok().flatten().unwrap_or_default(),
            s3_config_field: 0,
            s3_config_editing: false,
            s3_config_input: String::new(),
            s3_config_message: None,
            add_machine_open: false,
            add_machine_name: String::new(),
            add_machine_user: String::new(),
            add_machine_host: String::new(),
            add_machine_focus: AddMachineField::default(),
            session_modal_open: false,
            session_modal_instance: None,
            session_python_version: "3.12".to_string(),
            session_repo_path: String::new(),
            session_command: String::new(),
            session_skip_tmux: false,
            session_modal_focus: SessionModalField::default(),
            setup_rx: None,
            compared_runs: Vec::new(),
            starred_runs: load_starred_runs(),
            run_notes: load_run_notes(),
            compare_focused: false,
            s3_pull_rx: None,
            s3_pull_status: None,
            s3_pull_time: None,
            cached_example: None,
            goto_step_input: None,
            pull_run_modal_open: false,
            note_modal_open: false,
            note_modal_input: String::new(),
            note_modal_run_name: String::new(),
            move_run_modal_open: false,
            move_run_input: String::new(),
            move_run_index: None,
            pull_run_input: String::new(),
            config_cursor: 0,
            config_copied_at: None,
            selected_checkpoint: 0,
            checkpoint_download_modal: false,
            checkpoint_download_options: Vec::new(),
            checkpoint_download_selected: 0,
            checkpoint_download_step: 0,
            show_system_metrics: false,
            timestamp_x_metrics: HashSet::new(),
            config_filters: Vec::new(),
            filter_modal_open: false,
            filter_modal_phase: FilterPhase::default(),
            filter_modal_input: String::new(),
            filter_modal_selected: 0,
            filter_modal_key: String::new(),
            search_query: String::new(),
            search_editing: false,
            hide_completed: false,
            archived_projects: load_archived_projects(),
            show_archived: false,
            artifacts: load_artifacts_from_cache(),
            selected_artifact: 0,
            artifact_detail_scroll: 0,
            pending_delete_artifact: false,
            pending_delete_checkpoint: false,
            help_overlay_open: false,
            help_overlay_scroll: 0,
            confusion_step_idx: HashMap::new(),
        }
    }

    fn update_size(&mut self, width: u16, height: u16) {
        self.term_width = width;
        self.term_height = height;
        self.ensure_card_visible();
    }

    // Build the flattened list of entries (projects and runs)
    fn list_entries(&self) -> Vec<ListEntry> {
        use std::collections::BTreeMap;

        // Group runs by project
        let mut projects: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        let search_lower = self.search_query.to_lowercase();

        for (i, run) in self.runs.iter().enumerate() {
            if self.hide_completed && !run.is_running() {
                continue;
            }
            if !self.config_filters.is_empty() && !run_matches_filters(run, &self.config_filters) {
                continue;
            }
            if !search_lower.is_empty() && !run.name.to_lowercase().contains(&search_lower) {
                continue;
            }
            let project = run
                .project
                .clone()
                .unwrap_or_else(|| "(no project)".to_string());
            if !self.show_archived && self.archived_projects.contains(&project) {
                continue;
            }
            projects.entry(project).or_default().push(i);
        }

        let mut entries = Vec::new();

        for (project_name, run_indices) in projects {
            entries.push(ListEntry::Project {
                name: project_name.clone(),
            });

            if !search_lower.is_empty() || self.expanded_projects.contains(&project_name) {
                let mut starred: Vec<usize> = Vec::new();
                let mut unstarred: Vec<usize> = Vec::new();
                for run_index in run_indices {
                    if self
                        .starred_runs
                        .contains(&self.runs[run_index].display_name())
                    {
                        starred.push(run_index);
                    } else {
                        unstarred.push(run_index);
                    }
                }
                for run_index in starred.into_iter().chain(unstarred) {
                    entries.push(ListEntry::Run { run_index });
                }
            }
        }

        entries
    }

    fn sync_list_selection_to_run(&mut self) {
        if let Some(project) = self.runs.get(self.selected_run).map(|r| {
            r.project
                .clone()
                .unwrap_or_else(|| "(no project)".to_string())
        }) {
            self.expanded_projects.insert(project);
        }
        let entries = self.list_entries();
        if let Some(pos) = entries.iter().position(|e| match e {
            ListEntry::Run { run_index } => *run_index == self.selected_run,
            _ => false,
        }) {
            self.selected_list_item = pos;
            self.list_state.select(Some(pos));
        }
    }

    fn refresh_runs(&mut self) {
        let current_name = self.runs.get(self.selected_run).map(|r| r.name.clone());

        let compared_paths: Vec<PathBuf> = self
            .compared_runs
            .iter()
            .filter_map(|&idx| self.runs.get(idx).map(|r| r.path.clone()))
            .collect();

        let mut old_runs = std::mem::take(&mut self.runs);
        self.runs = load_runs_lightweight();

        for new_run in &mut self.runs {
            if let Some(old_idx) = old_runs.iter().position(|r| r.path == new_run.path) {
                let old_run = old_runs.swap_remove(old_idx);
                if old_run.data_loaded {
                    new_run.metrics = old_run.metrics;
                    new_run.examples = old_run.examples;
                    new_run.confusion_matrices = old_run.confusion_matrices;
                    new_run.checkpoints = old_run.checkpoints;
                    new_run.data_loaded = true;
                    new_run.data_loaded_at = old_run.data_loaded_at;
                }
            }
        }

        if let Some(name) = current_name {
            if let Some(idx) = self.runs.iter().position(|r| r.name == name) {
                self.selected_run = idx;
            } else {
                self.selected_run = self.selected_run.min(self.runs.len().saturating_sub(1));
            }
        }

        self.compared_runs = compared_paths
            .iter()
            .filter_map(|path| self.runs.iter().position(|r| r.path == *path))
            .collect();

        let entries = self.list_entries();
        self.selected_list_item = self.selected_list_item.min(entries.len().saturating_sub(1));
        self.list_state.select(Some(self.selected_list_item));
    }

    fn refresh_artifacts(&mut self) {
        self.artifacts = load_artifacts_from_cache();
        self.selected_artifact = self
            .selected_artifact
            .min(self.artifacts.len().saturating_sub(1));
    }

    fn start_s3_pull_artifacts(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some("Pulling artifacts...".to_string());

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let dest = artifacts_dir();
                let _ = tx.send(S3PullMessage::Pulling("Pulling artifacts...".to_string()));
                match client.download_all_artifact_metas(&dest).await {
                    Ok(count) => {
                        let _ =
                            tx.send(S3PullMessage::Done(format!("Pulled {} artifact(s)", count)));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Pull failed: {}", e)));
                    }
                }
            });
        });
    }

    fn start_s3_delete_artifact(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let Some(artifact) = self.artifacts.get(self.selected_artifact) else {
            return;
        };

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let name = artifact.name.clone();
        let local_dir = artifacts_dir().join(&name);

        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Deleting {}...", name));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                match client.delete_artifact(&name).await {
                    Ok(()) => {
                        let _ = std::fs::remove_dir_all(&local_dir);
                        let _ = tx.send(S3PullMessage::Done(format!("Deleted {}", name)));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Delete failed: {}", e)));
                    }
                }
            });
        });
    }

    fn start_s3_delete_checkpoint(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let Some(run) = self.current_run() else {
            return;
        };

        let Some(ckpt) = run.checkpoints.get(self.selected_checkpoint) else {
            return;
        };

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let step = ckpt.step;
        let project = run.project.clone().unwrap_or_default();
        let run_name = run.name.clone();
        let run_path = run.path.clone();

        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Deleting checkpoint step {}...", step));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                match client.delete_checkpoint(&project, &run_name, step).await {
                    Ok(()) => {
                        let local_dir = run_path.join("checkpoints").join(step.to_string());
                        let _ = std::fs::remove_dir_all(&local_dir);

                        let ckpt_json = run_path.join("checkpoints.json");
                        if let Ok(content) = std::fs::read_to_string(&ckpt_json)
                            && let Ok(mut entries) =
                                serde_json::from_str::<Vec<serde_json::Value>>(&content)
                        {
                            entries
                                .retain(|e| e.get("step").and_then(|v| v.as_u64()) != Some(step));
                            if let Ok(updated) = serde_json::to_string_pretty(&entries) {
                                let _ = std::fs::write(&ckpt_json, updated);
                            }
                        }

                        let _ = tx.send(S3PullMessage::Done(format!(
                            "Deleted checkpoint step {}",
                            step
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Delete failed: {}", e)));
                    }
                }
            });
        });
    }

    fn start_s3_download_artifact(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let Some(artifact) = self.artifacts.get(self.selected_artifact) else {
            return;
        };

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let name = artifact.name.clone();
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Downloading {}...", name));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let dest = artifacts_dir();
                let _ = tx.send(S3PullMessage::Pulling(format!("Downloading {}...", name)));
                match client.download_artifact_data(&name, &dest).await {
                    Ok(bytes) => {
                        let size = if bytes >= 1_048_576 {
                            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
                        } else if bytes >= 1024 {
                            format!("{:.1} KB", bytes as f64 / 1024.0)
                        } else {
                            format!("{} B", bytes)
                        };
                        let _ = tx.send(S3PullMessage::Done(format!(
                            "Downloaded {} ({})",
                            name, size
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Download failed: {}", e)));
                    }
                }
            });
        });
    }

    fn ensure_run_loaded(&mut self, idx: usize) {
        let needs_load = if let Some(run) = self.runs.get(idx) {
            if !run.data_loaded {
                true
            } else if run.is_running() {
                run.data_loaded_at
                    .map(|t| t.elapsed() > Duration::from_secs(30))
                    .unwrap_or(true)
            } else {
                false
            }
        } else {
            false
        };
        if needs_load {
            let path = self.runs[idx].path.clone();
            if let Some(updated) = data::reload_run(&path) {
                self.runs[idx] = updated;
            }
        }
    }

    fn start_s3_pull(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let Some(run) = self.current_run() else {
            return;
        };

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let project = run.project.clone().unwrap_or_default();
        let name = run.name.clone();
        let display = run.display_name();
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Pulling {}...", display));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let runs_dir = runs_dir();
                let _ = tx.send(S3PullMessage::Pulling(format!(
                    "Pulling {}/{}...",
                    project, name
                )));
                match client
                    .download_run(&project, &name, &runs_dir, false, false)
                    .await
                {
                    Ok(false) => {
                        let _ = tx.send(S3PullMessage::Done(format!(
                            "{}/{} already completed",
                            project, name
                        )));
                    }
                    Ok(true) => {
                        let _ =
                            tx.send(S3PullMessage::Done(format!("Pulled {}/{}", project, name)));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Pull failed: {}", e)));
                    }
                }
            });
        });
    }

    fn start_s3_pull_checkpoint(&mut self, step: u64, files: Option<Vec<String>>) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let Some(run) = self.current_run() else {
            return;
        };

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let project = run.project.clone().unwrap_or_default();
        let name = run.name.clone();
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);

        let desc = match &files {
            Some(f) => format!("Downloading {} step {}...", f.join(", "), step),
            None => format!("Downloading checkpoint step {}...", step),
        };
        self.s3_pull_status = Some(desc.clone());

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let runs_dir = runs_dir();
                let _ = tx.send(S3PullMessage::Pulling(desc));

                let file_refs: Option<Vec<&str>> = files
                    .as_ref()
                    .map(|f| f.iter().map(|s| s.as_str()).collect());
                let file_slices: Option<&[&str]> = file_refs.as_deref();

                match client
                    .download_checkpoint(&project, &name, step, &runs_dir, file_slices)
                    .await
                {
                    Ok(()) => {
                        let _ = tx.send(S3PullMessage::Done(format!(
                            "Downloaded checkpoint step {}",
                            step
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!(
                            "Checkpoint download failed: {}",
                            e
                        )));
                    }
                }
            });
        });
    }

    fn start_s3_pull_compared(&mut self) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let runs: Vec<(String, String)> = self
            .compared_runs
            .iter()
            .filter_map(|&idx| self.runs.get(idx))
            .map(|r| (r.project.clone().unwrap_or_default(), r.name.clone()))
            .collect();

        if runs.is_empty() {
            return;
        }

        let count = runs.len();
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Pulling {} runs...", count));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let runs_dir = runs_dir();
                let mut skipped = 0usize;
                for (i, (project, name)) in runs.iter().enumerate() {
                    let _ = tx.send(S3PullMessage::Pulling(format!(
                        "Pulling {}/{} ({}/{})...",
                        project,
                        name,
                        i + 1,
                        count
                    )));
                    match client
                        .download_run(project, name, &runs_dir, false, false)
                        .await
                    {
                        Ok(false) => skipped += 1,
                        Ok(true) => {}
                        Err(e) => {
                            let _ = tx.send(S3PullMessage::Error(format!("Pull failed: {}", e)));
                            return;
                        }
                    }
                }
                let msg = if skipped > 0 {
                    format!(
                        "Pulled {} runs ({} skipped, already completed)",
                        count, skipped
                    )
                } else {
                    format!("Pulled {} runs", count)
                };
                let _ = tx.send(S3PullMessage::Done(msg));
            });
        });
    }

    fn start_s3_pull_project(&mut self, project: &str) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let project = project.to_string();
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Listing runs for {}...", project));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let remote_runs = match client.list_runs(Some(&project)).await {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("List failed: {}", e)));
                        return;
                    }
                };
                if remote_runs.is_empty() {
                    let _ = tx.send(S3PullMessage::Done(format!(
                        "No remote runs for {}",
                        project
                    )));
                    return;
                }
                let count = remote_runs.len();
                let runs_dir = runs_dir();
                let mut skipped = 0usize;
                for (i, rr) in remote_runs.iter().enumerate() {
                    let _ = tx.send(S3PullMessage::Pulling(format!(
                        "Pulling {}/{} ({}/{})...",
                        project,
                        rr.name,
                        i + 1,
                        count
                    )));
                    match client
                        .download_run(&project, &rr.name, &runs_dir, false, false)
                        .await
                    {
                        Ok(false) => skipped += 1,
                        Ok(true) => {}
                        Err(e) => {
                            let _ = tx.send(S3PullMessage::Error(format!("Pull failed: {}", e)));
                            return;
                        }
                    }
                }
                let msg = if skipped > 0 {
                    format!(
                        "Pulled {} runs for {} ({} skipped, already completed)",
                        count, project, skipped
                    )
                } else {
                    format!("Pulled {} runs for {}", count, project)
                };
                let _ = tx.send(S3PullMessage::Done(msg));
            });
        });
    }

    fn start_s3_pull_by_name(&mut self, input: &str) {
        if self.s3_pull_rx.is_some() {
            return;
        }

        let parts: Vec<&str> = input.splitn(2, '/').collect();
        let (project, name) = match parts.as_slice() {
            [project, name] => (project.trim().to_string(), name.trim().to_string()),
            _ => {
                self.s3_pull_status = Some("Format: project/run".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        if project.is_empty() || name.is_empty() {
            self.s3_pull_status = Some("Format: project/run".to_string());
            self.s3_pull_time = Some(Instant::now());
            return;
        }

        let config = match s3::load_config() {
            Ok(Some(config)) => config,
            _ => {
                self.s3_pull_status = Some("S3 not configured".to_string());
                self.s3_pull_time = Some(Instant::now());
                return;
            }
        };

        let display = format!("{}/{}", project, name);
        let (tx, rx) = mpsc::channel();
        self.s3_pull_rx = Some(rx);
        self.s3_pull_status = Some(format!("Pulling {}...", display));

        thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send(S3PullMessage::Error(format!("Runtime error: {}", e)));
                    return;
                }
            };
            runtime.block_on(async {
                let client = match s3::S3Client::new(config).await {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("S3 error: {}", e)));
                        return;
                    }
                };
                let runs_dir = runs_dir();
                let _ = tx.send(S3PullMessage::Pulling(format!(
                    "Pulling {}/{}...",
                    project, name
                )));
                match client
                    .download_run(&project, &name, &runs_dir, false, false)
                    .await
                {
                    Ok(false) => {
                        let _ = tx.send(S3PullMessage::Done(format!(
                            "{}/{} already completed",
                            project, name
                        )));
                    }
                    Ok(true) => {
                        let _ =
                            tx.send(S3PullMessage::Done(format!("Pulled {}/{}", project, name)));
                    }
                    Err(e) => {
                        let _ = tx.send(S3PullMessage::Error(format!("Pull failed: {}", e)));
                    }
                }
            });
        });
    }

    fn refresh_infra(&mut self) {
        self.infra_loading = true;
        self.infra_error = None;
        self.infra_instances.clear();

        let provider = self.selected_infra_provider;

        if provider == Provider::Local {
            self.infra_instances = self
                .infra_config
                .local
                .iter()
                .map(|m| m.to_instance())
                .collect();
        } else {
            let config = self.infra_config.get_provider_config(provider);
            if let Some(api_key) = &config.api_key {
                let client = get_provider(provider, api_key);
                match client.list_instances() {
                    Ok(instances) => {
                        self.infra_instances.extend(instances);
                    }
                    Err(e) => {
                        self.infra_error = Some(format!("{}: {}", provider.display_name(), e));
                    }
                }
            }
        }

        self.infra_loading = false;
        self.infra_last_refresh = Instant::now();
        self.selected_infra_instance = self
            .selected_infra_instance
            .min(self.infra_instances.len().saturating_sub(1));
    }

    fn refresh_infra_types(&mut self) {
        self.infra_loading = true;
        self.infra_error = None;
        self.infra_types.clear();

        let provider = self.selected_infra_provider;

        if provider == Provider::Local {
            self.infra_loading = false;
            return;
        }

        let config = self.infra_config.get_provider_config(provider);

        if let Some(api_key) = &config.api_key {
            let client = get_provider(provider, api_key);
            match client.list_instance_types() {
                Ok(types) => {
                    self.infra_types = types;
                }
                Err(e) => {
                    self.infra_error = Some(format!("{}: {}", provider.display_name(), e));
                }
            }
        } else {
            self.infra_error = Some(format!(
                "No API key configured for {}",
                provider.display_name()
            ));
        }

        self.infra_loading = false;
        self.sort_infra_types();
        self.selected_infra_type = self
            .selected_infra_type
            .min(self.infra_types.len().saturating_sub(1));
        self.infra_types_list_state
            .select(Some(self.selected_infra_type));
    }

    fn sort_infra_types(&mut self) {
        match self.infra_type_sort {
            InfraTypeSort::Name => self.infra_types.sort_by(|a, b| a.name.cmp(&b.name)),
            InfraTypeSort::Price => self
                .infra_types
                .sort_by(|a, b| a.price_cents_per_hour.cmp(&b.price_cents_per_hour)),
            InfraTypeSort::Vram => self
                .infra_types
                .sort_by(|a, b| b.gpu_memory_gib.cmp(&a.gpu_memory_gib)),
        }
    }

    fn terminate_instance(&mut self, index: usize) {
        if let Some(instance) = self.infra_instances.get(index) {
            let provider = instance.provider;
            let instance_id = instance.id.clone();
            let config = self.infra_config.get_provider_config(provider);

            if let Some(api_key) = &config.api_key {
                let client = get_provider(provider, api_key);
                if let Err(e) = client.terminate(&[instance_id]) {
                    self.infra_error = Some(format!("Terminate failed: {}", e));
                } else {
                    self.refresh_infra();
                }
            }
        }
    }

    fn launch_ssh(&self, instance: &Instance) -> Result<()> {
        let ip = instance
            .ip
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No IP address"))?;

        let ssh_cmd = if ip.contains(':') {
            let parts: Vec<&str> = ip.split(':').collect();
            format!(
                "ssh -o StrictHostKeyChecking=no -p {} {}@{}",
                parts[1], instance.ssh_user, parts[0]
            )
        } else {
            format!(
                "ssh -o StrictHostKeyChecking=no {}@{}",
                instance.ssh_user, ip
            )
        };

        #[cfg(target_os = "macos")]
        {
            let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
            let script = if term_program == "iTerm.app" {
                format!(
                    "tell application \"iTerm2\" to tell current window to create tab with default profile command \"{}\"",
                    ssh_cmd
                )
            } else {
                format!("tell application \"Terminal\" to do script \"{}\"", ssh_cmd)
            };
            std::process::Command::new("osascript")
                .args(["-e", &script])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()?;
        }

        #[cfg(target_os = "linux")]
        {
            std::process::Command::new("x-terminal-emulator")
                .args(["-e", &ssh_cmd])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()?;
        }

        Ok(())
    }

    fn do_launch_ssh_with_setup(&mut self, instance: &Instance) {
        let ip = match instance.ip.as_ref() {
            Some(ip) => ip.clone(),
            None => {
                self.infra_error = Some("No IP address available".to_string());
                return;
            }
        };

        let repo_path = self.session_repo_path.clone();
        if repo_path.is_empty() {
            self.infra_error = Some("No repository path specified".to_string());
            return;
        }

        let ssh_user = instance.ssh_user.clone();
        let instance_id = instance.id.clone();
        let provider_str = instance.provider.as_str().to_string();
        let python_version = self.session_python_version.clone();
        let command = self.session_command.clone();
        let skip_tmux = self.session_skip_tmux;
        let (tx, rx) = mpsc::channel();
        self.setup_rx = Some(rx);
        self.infra_error = Some("Syncing code...".to_string());
        self.infra_message_time = Some(Instant::now());

        thread::spawn(move || {
            let (host, port) = if ip.contains(':') {
                let parts: Vec<&str> = ip.split(':').collect();
                (parts[0].to_string(), Some(parts[1].to_string()))
            } else {
                (ip.clone(), None)
            };

            let ssh_port_args = port
                .as_ref()
                .map(|p| format!("-p {}", p))
                .unwrap_or_default();

            let ssh_base = if ssh_port_args.is_empty() {
                format!("ssh -o StrictHostKeyChecking=no {}@{}", ssh_user, host)
            } else {
                format!(
                    "ssh -o StrictHostKeyChecking=no {} {}@{}",
                    ssh_port_args, ssh_user, host
                )
            };

            let exttyignore_path = std::path::Path::new(&repo_path).join(".exttyignore");

            let mut rsync_cmd = std::process::Command::new("rsync");
            rsync_cmd.args(["-az", "--info=progress2", "--no-inc-recursive"]);

            if exttyignore_path.exists() {
                let filter_arg = format!("merge {}", exttyignore_path.display());
                rsync_cmd.args(["--filter", &filter_arg]);
            }

            rsync_cmd
                .args([
                    "--exclude",
                    ".git",
                    "--exclude",
                    "__pycache__",
                    "--exclude",
                    ".venv",
                    "--exclude",
                    "*.pyc",
                    "--exclude",
                    ".mypy_cache",
                    "--exclude",
                    "*.egg-info",
                ])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());

            if let Some(p) = &port {
                rsync_cmd.args(["-e", &format!("ssh -o StrictHostKeyChecking=no -p {}", p)]);
            } else {
                rsync_cmd.args(["-e", "ssh -o StrictHostKeyChecking=no"]);
            }

            let repo_with_slash = if repo_path.ends_with('/') {
                repo_path.clone()
            } else {
                format!("{}/", repo_path)
            };

            let remote_dir = project_remote_dir(std::path::Path::new(&repo_path));

            let mkdir_result = std::process::Command::new("ssh")
                .arg("-o")
                .arg("StrictHostKeyChecking=no")
                .args(
                    port.as_ref()
                        .map(|p| vec!["-p", p.as_str()])
                        .unwrap_or_default(),
                )
                .arg(format!("{}@{}", ssh_user, host))
                .arg(format!("mkdir -p $HOME/{}", remote_dir))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            if let Ok(status) = mkdir_result
                && !status.success()
            {
                let _ = tx.send(SetupMessage::Error(
                    "Failed to create remote project directory".to_string(),
                ));
                return;
            }

            rsync_cmd
                .arg(&repo_with_slash)
                .arg(format!("{}@{}:{}/", ssh_user, host, remote_dir));

            let rsync_result = match rsync_cmd.spawn() {
                Ok(mut child) => {
                    if let Some(stdout) = child.stdout.take() {
                        use std::io::{BufRead, BufReader};
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().map_while(Result::ok) {
                            if let Some(pct_start) = line.find('%')
                                && let Some(num_start) =
                                    line[..pct_start].rfind(char::is_whitespace)
                                && let Ok(pct) =
                                    line[num_start + 1..pct_start].trim().parse::<u32>()
                            {
                                let _ = tx.send(SetupMessage::Status(format!(
                                    "Syncing code... {}%",
                                    pct
                                )));
                            }
                        }
                    }
                    child.wait()
                }
                Err(e) => Err(e),
            };

            match rsync_result {
                Ok(status) if status.success() => {
                    let _ = tx.send(SetupMessage::Status(
                        "Code synced, copying config...".to_string(),
                    ));
                }
                Ok(status) => {
                    let _ = tx.send(SetupMessage::Error(format!(
                        "rsync failed (exit {})",
                        status
                    )));
                    return;
                }
                Err(e) => {
                    let _ = tx.send(SetupMessage::Error(format!("rsync failed: {}", e)));
                    return;
                }
            }

            let s3_config_path = dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".extty")
                .join("s3")
                .join("config.toml");

            if s3_config_path.exists() {
                let mut scp_cmd = std::process::Command::new("bash");
                scp_cmd
                    .arg("-c")
                    .arg(format!(
                        "{} 'mkdir -p ~/.extty/s3' && scp -o StrictHostKeyChecking=no {} {} {}@{}:~/.extty/s3/config.toml",
                        ssh_base,
                        port.as_ref().map(|p| format!("-P {}", p)).unwrap_or_default(),
                        s3_config_path.display(),
                        ssh_user,
                        host,
                    ))
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());

                if let Ok(status) = scp_cmd.status()
                    && !status.success()
                {
                    let _ = tx.send(SetupMessage::Status(
                        "Warning: failed to copy S3 config".to_string(),
                    ));
                }
            }

            let git_hash = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&repo_path)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .and_then(|output| {
                    if output.status.success() {
                        String::from_utf8(output.stdout)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                });

            let _ = tx.send(SetupMessage::Status("Uploading script...".to_string()));

            let script = generate_script(&ScriptOptions {
                python_version: &python_version,
                command: &command,
                skip_tmux,
                project_dir: &remote_dir,
                git_hash: git_hash.as_deref(),
                run_command: &command,
                instance_id: &instance_id,
                provider: &provider_str,
            });

            let upload_status = std::process::Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "{} 'cat > /tmp/extty_bootstrap.sh && chmod +x /tmp/extty_bootstrap.sh'",
                    ssh_base
                ))
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .and_then(|mut child| {
                    use std::io::Write;
                    if let Some(stdin) = child.stdin.as_mut() {
                        stdin.write_all(script.as_bytes())?;
                    }
                    child.wait()
                });

            match upload_status {
                Ok(status) if status.success() => {}
                Ok(status) => {
                    let _ = tx.send(SetupMessage::Error(format!(
                        "Script upload failed: exit {}",
                        status
                    )));
                    return;
                }
                Err(e) => {
                    let _ = tx.send(SetupMessage::Error(format!("Script upload failed: {}", e)));
                    return;
                }
            }

            let ssh_cmd = format!("{} -t 'bash /tmp/extty_bootstrap.sh'", ssh_base);

            #[cfg(target_os = "macos")]
            {
                let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
                let applescript = if term_program == "iTerm.app" {
                    format!(
                        "tell application \"iTerm2\" to tell current window to create tab with default profile command \"{}\"",
                        ssh_cmd.replace('"', "\\\"")
                    )
                } else {
                    format!(
                        "tell application \"Terminal\" to do script \"{}\"",
                        ssh_cmd.replace('"', "\\\"")
                    )
                };
                let _ = std::process::Command::new("osascript")
                    .args(["-e", &applescript])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }

            #[cfg(target_os = "linux")]
            {
                let _ = std::process::Command::new("x-terminal-emulator")
                    .args(["-e", "bash", "-c", &ssh_cmd])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }

            let _ = tx.send(SetupMessage::Done("SSH session launched".to_string()));
        });
    }

    fn filtered_infra_instances(&self) -> Vec<&Instance> {
        self.infra_instances
            .iter()
            .filter(|i| i.provider == self.selected_infra_provider)
            .collect()
    }

    fn grid_layout(&self) -> (usize, usize) {
        // Returns (visible_rows, cols) for the card grid
        let card_width = 40u16;
        let card_height = 12u16;
        let grid_height = self.term_height.saturating_sub(5); // header + footer

        let config_panel_width = if self.view == View::Compare {
            40u16 * self.compared_runs.len() as u16
        } else {
            35u16
        };
        let has_config = match self.view {
            View::Compare => self
                .compared_runs
                .iter()
                .any(|&ri| self.runs.get(ri).and_then(|r| r.config.as_ref()).is_some()),
            _ => self
                .current_run()
                .map(|r| r.config.is_some())
                .unwrap_or(false),
        };
        let effective_width = if self.show_config && has_config {
            self.term_width.saturating_sub(config_panel_width)
        } else {
            self.term_width
        };

        let cols = (effective_width / card_width).max(1) as usize;
        let visible_rows = (grid_height / card_height).max(1) as usize;
        (visible_rows, cols)
    }

    fn ensure_card_visible(&mut self) {
        let (visible_rows, cols) = self.grid_layout();
        let card_count = match self.view {
            View::Compare => self.compare_cards().len(),
            View::Focused if self.compare_focused => self.compare_cards().len(),
            _ => self.card_count(),
        };
        self.selected_card = self.selected_card.min(card_count.saturating_sub(1));
        let card_row = self.selected_card / cols;
        let total_rows = card_count.div_ceil(cols);
        let max_scroll = total_rows.saturating_sub(visible_rows);
        if card_row < self.scroll_offset {
            self.scroll_offset = card_row;
        } else if card_row >= self.scroll_offset + visible_rows {
            self.scroll_offset = (card_row + 1).saturating_sub(visible_rows);
        }
        self.scroll_offset = self.scroll_offset.min(max_scroll);
    }

    fn current_run(&self) -> Option<&Run> {
        self.runs.get(self.selected_run)
    }

    fn run_artifacts(&self, run: &Run) -> Vec<&Artifact> {
        let run_project = run.project.as_deref().unwrap_or("");
        self.artifacts
            .iter()
            .filter(|a| {
                a.run_project.as_deref() == Some(run_project)
                    && a.run_name.as_deref() == Some(&run.name)
            })
            .collect()
    }

    fn chart_x_axis(&self, metric_name: &str) -> XAxis {
        if self.timestamp_x_metrics.contains(metric_name) {
            XAxis::Timestamp
        } else {
            XAxis::Step
        }
    }

    fn toggle_chart_x_axis(&mut self, metric_name: &str) {
        if !self.timestamp_x_metrics.remove(metric_name) {
            self.timestamp_x_metrics.insert(metric_name.to_string());
        }
    }

    fn cards(&self) -> Vec<Card> {
        let Some(run) = self.current_run() else {
            return vec![];
        };

        let mut cards = Vec::new();

        let mut metric_names: Vec<&String> = run
            .metrics
            .keys()
            .filter(|n| {
                if self.show_system_metrics {
                    n.starts_with("sys/")
                } else {
                    !n.starts_with("sys/")
                }
            })
            .collect();
        metric_names.sort();
        for name in metric_names {
            cards.push(Card::Chart { name: name.clone() });
        }

        if !self.show_system_metrics {
            let mut example_names: Vec<&String> = run.examples.keys().collect();
            example_names.sort();
            for name in example_names {
                cards.push(Card::Examples { name: name.clone() });
            }

            let mut cm_names: Vec<&String> = run.confusion_matrices.keys().collect();
            cm_names.sort();
            for name in cm_names {
                cards.push(Card::ConfusionMatrix { name: name.clone() });
            }

            if !run.checkpoints.is_empty() {
                cards.push(Card::Checkpoints);
            }

            if !self.run_artifacts(run).is_empty() {
                cards.push(Card::Artifacts);
            }
        }

        cards
    }

    fn compare_cards(&self) -> Vec<Card> {
        let show_sys = self.show_system_metrics;
        let mut metric_names: Vec<String> = self
            .compared_runs
            .iter()
            .filter_map(|&idx| self.runs.get(idx))
            .flat_map(|run| run.metrics.keys().cloned())
            .filter(|n| {
                if show_sys {
                    n.starts_with("sys/")
                } else {
                    !n.starts_with("sys/")
                }
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        metric_names.sort();

        let mut cards: Vec<Card> = metric_names
            .into_iter()
            .map(|name| Card::Chart { name })
            .collect();

        if !show_sys {
            let mut example_names: Vec<String> = self
                .compared_runs
                .iter()
                .filter_map(|&idx| self.runs.get(idx))
                .flat_map(|run| run.examples.keys().cloned())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            example_names.sort();

            cards.extend(
                example_names
                    .into_iter()
                    .map(|name| Card::Examples { name }),
            );
        }
        cards
    }

    fn compare_example_steps(&self, name: &str) -> Vec<u64> {
        let steps: BTreeSet<u64> = self
            .compared_runs
            .iter()
            .filter_map(|&idx| self.runs.get(idx))
            .filter_map(|run| run.examples.get(name))
            .flat_map(|g| g.steps())
            .collect();
        steps.into_iter().collect()
    }

    fn card_count(&self) -> usize {
        let Some(run) = self.current_run() else {
            return 0;
        };
        let has_checkpoints = if run.checkpoints.is_empty() { 0 } else { 1 };
        let has_artifacts = if self.run_artifacts(run).is_empty() {
            0
        } else {
            1
        };
        run.metrics.len()
            + run.examples.len()
            + run.confusion_matrices.len()
            + has_checkpoints
            + has_artifacts
    }

    fn get_or_load_example(&mut self, name: &str, idx: usize) -> Option<&Example> {
        if self
            .cached_example
            .as_ref()
            .is_none_or(|(n, i, _)| n != name || *i != idx)
        {
            let example = self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .and_then(|g| g.load(idx));
            if let Some(ex) = example {
                self.cached_example = Some((name.to_string(), idx, ex));
            } else {
                return None;
            }
        }
        self.cached_example.as_ref().map(|(_, _, ex)| ex)
    }

    // Calculate wrapped line count for text given a width
    fn wrapped_line_count(text: &str, width: usize) -> usize {
        if width == 0 {
            return 0;
        }
        let mut count = 0;
        for line in text.lines() {
            if line.is_empty() {
                count += 1;
            } else {
                // Estimate wrapped lines (chars / width, rounded up)
                count += line.chars().count().div_ceil(width)
            }
        }
        // Handle case where text doesn't end with newline but has content
        if count == 0 && !text.is_empty() {
            count = 1;
        }
        count
    }

    fn active_cards(&self) -> Vec<Card> {
        match self.view {
            View::RunDetail | View::List => self.cards(),
            View::Compare => self.compare_cards(),
            View::Focused if self.compare_focused => self.compare_cards(),
            View::Focused => match self.view_mode {
                ViewMode::Runs => self.cards(),
                ViewMode::Artifacts | ViewMode::Infra => vec![],
            },
            View::ConfigFull
            | View::InfraList
            | View::InfraConfig
            | View::S3Config
            | View::ArtifactDetail => vec![],
        }
    }

    fn current_example_has_groundtruth(&self) -> bool {
        let cards = self.active_cards();
        match cards.get(self.selected_card) {
            Some(Card::Examples { name }) if self.compare_focused => {
                let steps = self.compare_example_steps(name);
                let step = steps.get(self.selected_example).copied();
                step.map(|step| {
                    self.compared_runs
                        .iter()
                        .filter_map(|&idx| self.runs.get(idx))
                        .any(|run| {
                            run.examples
                                .get(name)
                                .and_then(|g| g.find_step(step))
                                .and_then(|idx| run.examples.get(name).unwrap().load(idx))
                                .and_then(|ex| ex.groundtruth)
                                .is_some()
                        })
                })
                .unwrap_or(false)
            }
            Some(Card::Examples { name }) => {
                if let Some((ref cn, ci, ref ex)) = self.cached_example {
                    cn == name && ci == self.selected_example && ex.groundtruth.is_some()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    // Get max scroll offset for a section in focused view
    fn focused_max_scroll(&self, section: FocusedSection) -> usize {
        let cards = self.active_cards();
        let current_card = cards.get(self.selected_card);
        let has_gt = self.current_example_has_groundtruth();

        let total_height = self.term_height.saturating_sub(6) as usize;
        let visible_height = if has_gt {
            match section {
                FocusedSection::Prompt => (total_height * 30 / 100).saturating_sub(2),
                FocusedSection::Groundtruth | FocusedSection::Response => {
                    (total_height * 35 / 100).saturating_sub(2)
                }
            }
        } else {
            match section {
                FocusedSection::Prompt => (total_height * 40 / 100).saturating_sub(2),
                FocusedSection::Response => (total_height * 60 / 100).saturating_sub(2),
                FocusedSection::Groundtruth => 0,
            }
        };

        let wrap_width = self.term_width.saturating_sub(4) as usize;

        let text = match current_card {
            Some(Card::Examples { name }) => {
                if let Some((ref cn, ci, ref ex)) = self.cached_example {
                    if cn == name && ci == self.selected_example {
                        match section {
                            FocusedSection::Prompt => ex
                                .prompts
                                .get(self.selected_prompt)
                                .cloned()
                                .unwrap_or_default(),
                            FocusedSection::Groundtruth => ex
                                .groundtruth
                                .as_ref()
                                .and_then(|gt| gt.get(self.selected_prompt))
                                .cloned()
                                .unwrap_or_default(),
                            FocusedSection::Response => ex
                                .responses
                                .get(self.selected_prompt)
                                .and_then(|r| r.get(self.selected_response))
                                .cloned()
                                .unwrap_or_default(),
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        };

        let line_count = Self::wrapped_line_count(&text, wrap_width);
        line_count.saturating_sub(visible_height)
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let code = key.code;
        let modifiers = key.modifiers;

        if code == KeyCode::Char('c') && modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }

        if self.help_overlay_open {
            match code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?') => {
                    self.help_overlay_open = false;
                    self.help_overlay_scroll = 0;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.help_overlay_scroll = self.help_overlay_scroll.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.help_overlay_scroll = self.help_overlay_scroll.saturating_add(1);
                }
                KeyCode::PageUp => {
                    self.help_overlay_scroll = self.help_overlay_scroll.saturating_sub(10);
                }
                KeyCode::PageDown => {
                    self.help_overlay_scroll = self.help_overlay_scroll.saturating_add(10);
                }
                _ => {}
            }
            return;
        }

        if self.filter_modal_open {
            self.handle_filter_modal_key(code);
            return;
        }

        if self.checkpoint_download_modal {
            match code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    self.checkpoint_download_modal = false;
                }
                KeyCode::Up if self.checkpoint_download_selected > 0 => {
                    self.checkpoint_download_selected -= 1;
                }
                KeyCode::Down
                    if self.checkpoint_download_selected
                        < self.checkpoint_download_options.len().saturating_sub(1) =>
                {
                    self.checkpoint_download_selected += 1;
                }
                KeyCode::Enter => {
                    let step = self.checkpoint_download_step;
                    let selected =
                        &self.checkpoint_download_options[self.checkpoint_download_selected];
                    let files = if selected == "All files" {
                        let all: Vec<String> = self
                            .checkpoint_download_options
                            .iter()
                            .filter(|s| s.as_str() != "All files")
                            .cloned()
                            .collect();
                        Some(all)
                    } else {
                        Some(vec![selected.clone()])
                    };
                    self.checkpoint_download_modal = false;
                    self.start_s3_pull_checkpoint(step, files);
                }
                _ => {}
            }
            return;
        }

        if self.note_modal_open {
            self.handle_note_modal_key(code);
            return;
        }

        if self.move_run_modal_open {
            self.handle_move_run_modal_key(code);
            return;
        }

        if self.pull_run_modal_open {
            self.handle_pull_run_modal_key(code);
            return;
        }
        if self.session_modal_open {
            self.handle_session_modal_key(code);
            return;
        }
        if self.show_delete_confirm {
            self.handle_delete_confirm_key(code);
            return;
        }
        if self.show_terminate_confirm {
            self.handle_terminate_confirm_key(code);
            return;
        }
        if self.show_complete_confirm {
            self.handle_complete_confirm_key(code);
            return;
        }

        if let Some(ref mut input) = self.goto_step_input {
            match code {
                KeyCode::Char(c) if c.is_ascii_digit() => input.push(c),
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Esc => self.goto_step_input = None,
                KeyCode::Enter => {
                    let target: u64 = input.parse().unwrap_or(0);
                    self.goto_step_input = None;

                    let steps: Vec<u64> = if self.compare_focused {
                        let cards = self.compare_cards();
                        match cards.get(self.selected_card) {
                            Some(Card::Examples { name }) => self.compare_example_steps(name),
                            _ => vec![],
                        }
                    } else {
                        match self.view_mode {
                            ViewMode::Runs => {
                                let cards = self.cards();
                                match cards.get(self.selected_card) {
                                    Some(Card::Examples { name }) => self
                                        .current_run()
                                        .and_then(|r| r.examples.get(name))
                                        .map(|g| g.steps().collect())
                                        .unwrap_or_default(),
                                    _ => vec![],
                                }
                            }
                            ViewMode::Artifacts | ViewMode::Infra => vec![],
                        }
                    };

                    if !steps.is_empty() {
                        let best_idx = steps
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, s)| (**s as i64 - target as i64).unsigned_abs())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        self.selected_example = best_idx;
                        self.cached_example = None;
                        self.selected_prompt = 0;
                        self.selected_response = 0;
                        self.prompt_scroll_offset = 0;
                        self.response_scroll_offset = 0;
                        self.groundtruth_scroll_offset = 0;
                    }
                }
                _ => {}
            }
            return;
        }

        if self.search_editing {
            match code {
                KeyCode::Char(c) => {
                    self.search_query.push(c);
                    self.selected_list_item = 0;
                    self.list_state.select(Some(0));
                }
                KeyCode::Backspace => {
                    self.search_query.pop();
                    self.selected_list_item = 0;
                    self.list_state.select(Some(0));
                }
                KeyCode::Esc => {
                    self.search_query.clear();
                    self.search_editing = false;
                    self.selected_list_item = 0;
                    self.list_state.select(Some(0));
                }
                KeyCode::Enter => {
                    self.search_editing = false;
                }
                _ => {}
            }
            return;
        }

        if code == KeyCode::Char('?')
            && !self.add_machine_open
            && !self.launch_confirming
            && !self.launch_selecting_region
        {
            self.help_overlay_open = true;
            self.help_overlay_scroll = 0;
            return;
        }

        match self.view {
            View::List => match self.view_mode {
                ViewMode::Runs => self.handle_list_key(code),
                ViewMode::Artifacts => self.handle_artifacts_list_key(code),
                ViewMode::Infra => self.handle_infra_list_key(code, modifiers),
            },
            View::RunDetail => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_detail_key(code, visible_rows, cols);
            }
            View::Focused => self.handle_focused_key(key),
            View::ConfigFull => self.handle_config_full_key(code),
            View::Compare => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_compare_key(code, visible_rows, cols);
            }
            View::InfraList => self.handle_infra_list_key(code, modifiers),
            View::InfraConfig => self.handle_infra_config_key(code),
            View::S3Config => self.handle_s3_config_key(code),
            View::ArtifactDetail => self.handle_artifact_detail_key(code),
        }
    }

    fn handle_pull_run_modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.pull_run_modal_open = false;
                self.pull_run_input.clear();
            }
            KeyCode::Enter => {
                let input = self.pull_run_input.clone();
                self.pull_run_modal_open = false;
                self.pull_run_input.clear();
                self.start_s3_pull_by_name(&input);
            }
            KeyCode::Char(c) => {
                self.pull_run_input.push(c);
            }
            KeyCode::Backspace => {
                self.pull_run_input.pop();
            }
            _ => {}
        }
    }

    fn handle_note_modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.note_modal_open = false;
                self.note_modal_input.clear();
                self.note_modal_run_name.clear();
            }
            KeyCode::Enter => {
                let name = self.note_modal_run_name.clone();
                let input = self.note_modal_input.trim().to_string();
                self.note_modal_open = false;
                self.note_modal_input.clear();
                self.note_modal_run_name.clear();
                if input.is_empty() {
                    self.run_notes.remove(&name);
                } else {
                    self.run_notes.insert(name, input);
                }
                save_run_notes(&self.run_notes);
            }
            KeyCode::Char(c) => {
                self.note_modal_input.push(c);
            }
            KeyCode::Backspace => {
                self.note_modal_input.pop();
            }
            _ => {}
        }
    }

    fn handle_move_run_modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.move_run_modal_open = false;
                self.move_run_input.clear();
                self.move_run_index = None;
            }
            KeyCode::Enter => {
                let new_project = self.move_run_input.trim().to_string();
                if let Some(run_idx) = self.move_run_index
                    && let Some(run) = self.runs.get(run_idx)
                {
                    let old_display_name = run.display_name();
                    let old_path = run.path.clone();
                    let old_project = run.project.clone().unwrap_or_default();
                    let run_name = run.name.clone();

                    let new_display_name = if new_project.is_empty() {
                        run_name.clone()
                    } else {
                        format!("{}/{}", new_project, run_name)
                    };

                    if old_display_name == new_display_name {
                        self.move_run_modal_open = false;
                        self.move_run_input.clear();
                        self.move_run_index = None;
                        return;
                    }

                    if data::move_run(&old_path, &new_project).is_ok() {
                        if self.starred_runs.remove(&old_display_name) {
                            self.starred_runs.insert(new_display_name.clone());
                            save_starred_runs(&self.starred_runs);
                        }
                        if let Some(note) = self.run_notes.remove(&old_display_name) {
                            self.run_notes.insert(new_display_name.clone(), note);
                            save_run_notes(&self.run_notes);
                        }

                        if !old_project.is_empty()
                            && let Ok(Some(config)) = s3::load_config()
                        {
                            let new_proj = new_project.clone();
                            let run_n = run_name.clone();
                            let old_proj = old_project.clone();
                            let (tx, rx) = mpsc::channel();
                            self.s3_pull_rx = Some(rx);
                            self.s3_pull_status = Some(format!(
                                "Moving {} → {}...",
                                old_display_name, new_display_name
                            ));
                            thread::spawn(move || {
                                let runtime = match tokio::runtime::Runtime::new() {
                                    Ok(rt) => rt,
                                    Err(e) => {
                                        let _ = tx.send(S3PullMessage::Error(format!(
                                            "Runtime error: {}",
                                            e
                                        )));
                                        return;
                                    }
                                };
                                runtime.block_on(async {
                                    let client = match s3::S3Client::new(config).await {
                                        Ok(c) => c,
                                        Err(e) => {
                                            let _ = tx.send(S3PullMessage::Error(format!(
                                                "S3 error: {}",
                                                e
                                            )));
                                            return;
                                        }
                                    };
                                    match client.move_run(&old_proj, &new_proj, &run_n).await {
                                        Ok(()) => {
                                            let _ = tx.send(S3PullMessage::Done(format!(
                                                "Moved to {} on S3",
                                                if new_proj.is_empty() {
                                                    "(no project)".to_string()
                                                } else {
                                                    new_proj
                                                }
                                            )));
                                        }
                                        Err(e) => {
                                            let _ = tx.send(S3PullMessage::Error(format!(
                                                "S3 move failed: {}",
                                                e
                                            )));
                                        }
                                    }
                                });
                            });
                        }

                        self.refresh_runs();
                        if self.view == View::RunDetail {
                            self.view = View::List;
                        }
                    }
                }
                self.move_run_modal_open = false;
                self.move_run_input.clear();
                self.move_run_index = None;
            }
            KeyCode::Char(c) => {
                self.move_run_input.push(c);
            }
            KeyCode::Backspace => {
                self.move_run_input.pop();
            }
            _ => {}
        }
    }

    fn filtered_config_keys(&self) -> Vec<String> {
        let all = collect_config_keys(&self.runs);
        let query = self.filter_modal_input.to_lowercase();
        if query.is_empty() {
            all
        } else {
            all.into_iter()
                .filter(|k| k.to_lowercase().contains(&query))
                .collect()
        }
    }

    fn filtered_config_values(&self) -> Vec<String> {
        let all = collect_config_values(&self.runs, &self.filter_modal_key);
        let query = self.filter_modal_input.to_lowercase();
        if query.is_empty() {
            all
        } else {
            all.into_iter()
                .filter(|v| v.to_lowercase().contains(&query))
                .collect()
        }
    }

    fn handle_filter_modal_key(&mut self, code: KeyCode) {
        match self.filter_modal_phase {
            FilterPhase::KeySelect => match code {
                KeyCode::Esc => {
                    self.filter_modal_open = false;
                }
                KeyCode::Char('d') => {
                    let num_filters = self.config_filters.len();
                    if num_filters > 0 && self.filter_modal_selected < num_filters {
                        self.config_filters.remove(self.filter_modal_selected);
                        if self.filter_modal_selected >= self.config_filters.len()
                            && self.filter_modal_selected > 0
                        {
                            self.filter_modal_selected -= 1;
                        }
                        self.selected_list_item = 0;
                        self.list_state.select(Some(0));
                    } else {
                        self.filter_modal_input.push('d');
                        self.filter_modal_selected = 0;
                    }
                }
                KeyCode::Char(c) => {
                    self.filter_modal_input.push(c);
                    self.filter_modal_selected = 0;
                }
                KeyCode::Backspace => {
                    self.filter_modal_input.pop();
                    self.filter_modal_selected = 0;
                }
                KeyCode::Up => {
                    if self.filter_modal_selected > 0 {
                        self.filter_modal_selected -= 1;
                    }
                }
                KeyCode::Down => {
                    let count = self.config_filters.len() + self.filtered_config_keys().len();
                    if self.filter_modal_selected < count.saturating_sub(1) {
                        self.filter_modal_selected += 1;
                    }
                }
                KeyCode::Enter => {
                    let num_filters = self.config_filters.len();
                    let keys = self.filtered_config_keys();
                    if self.filter_modal_selected >= num_filters {
                        let key_idx = self.filter_modal_selected - num_filters;
                        if let Some(key) = keys.get(key_idx) {
                            self.filter_modal_key = key.clone();
                            self.filter_modal_phase = FilterPhase::ValueSelect;
                            self.filter_modal_input.clear();
                            self.filter_modal_selected = 0;
                        }
                    }
                }
                _ => {}
            },
            FilterPhase::ValueSelect => match code {
                KeyCode::Esc => {
                    self.filter_modal_phase = FilterPhase::KeySelect;
                    self.filter_modal_input.clear();
                    self.filter_modal_selected = 0;
                }
                KeyCode::Char(c) => {
                    self.filter_modal_input.push(c);
                    self.filter_modal_selected = 0;
                }
                KeyCode::Backspace => {
                    self.filter_modal_input.pop();
                    self.filter_modal_selected = 0;
                }
                KeyCode::Up => {
                    if self.filter_modal_selected > 0 {
                        self.filter_modal_selected -= 1;
                    }
                }
                KeyCode::Down => {
                    let count = self.filtered_config_values().len();
                    if self.filter_modal_selected < count.saturating_sub(1) {
                        self.filter_modal_selected += 1;
                    }
                }
                KeyCode::Enter => {
                    let values = self.filtered_config_values();
                    if let Some(val) = values.get(self.filter_modal_selected) {
                        self.config_filters
                            .push((self.filter_modal_key.clone(), val.clone()));
                        self.filter_modal_open = false;
                        self.selected_list_item = 0;
                        self.list_state.select(Some(0));
                    }
                }
                _ => {}
            },
        }
    }

    fn handle_delete_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some(run_idx) = self.pending_delete_run
                    && let Some(run) = self.runs.get(run_idx)
                {
                    let path = run.path.clone();
                    let project = run.project.clone().unwrap_or_default();
                    let name = run.name.clone();
                    let _ = data::delete_run(&path);

                    if let Ok(Some(config)) = s3::load_config() {
                        let (tx, rx) = mpsc::channel();
                        self.s3_pull_rx = Some(rx);
                        self.s3_pull_status = Some(format!("Deleting {}/{}...", project, name));
                        thread::spawn(move || {
                            let runtime = match tokio::runtime::Runtime::new() {
                                Ok(rt) => rt,
                                Err(e) => {
                                    let _ = tx.send(S3PullMessage::Error(format!(
                                        "Runtime error: {}",
                                        e
                                    )));
                                    return;
                                }
                            };
                            runtime.block_on(async {
                                let client = match s3::S3Client::new(config).await {
                                    Ok(c) => c,
                                    Err(e) => {
                                        let _ = tx
                                            .send(S3PullMessage::Error(format!("S3 error: {}", e)));
                                        return;
                                    }
                                };
                                match client.delete_run(&project, &name).await {
                                    Ok(()) => {
                                        let _ = tx.send(S3PullMessage::Done(format!(
                                            "Deleted {}/{} from S3",
                                            project, name
                                        )));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(S3PullMessage::Error(format!(
                                            "S3 delete failed: {}",
                                            e
                                        )));
                                    }
                                }
                            });
                        });
                    }

                    self.refresh_runs();
                    if self.view == View::RunDetail {
                        self.view = View::List;
                    }
                }
                if self.pending_delete_artifact {
                    self.start_s3_delete_artifact();
                    self.refresh_artifacts();
                    if self.view == View::ArtifactDetail {
                        self.view = View::List;
                    }
                }
                if self.pending_delete_checkpoint {
                    self.start_s3_delete_checkpoint();
                    self.refresh_runs();
                    self.selected_checkpoint = self.selected_checkpoint.min(
                        self.current_run()
                            .map(|r| r.checkpoints.len())
                            .unwrap_or(0)
                            .saturating_sub(1),
                    );
                }
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
                self.pending_delete_artifact = false;
                self.pending_delete_checkpoint = false;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
                self.pending_delete_artifact = false;
                self.pending_delete_checkpoint = false;
            }
            _ => {}
        }
    }

    fn handle_list_key(&mut self, code: KeyCode) {
        let entries = self.list_entries();
        let entry_count = entries.len();

        match code {
            KeyCode::Char('a') => {
                self.view_mode = ViewMode::Artifacts;
            }
            KeyCode::Char('i') => {
                self.view_mode = ViewMode::Infra;
                self.view = View::InfraList;
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('S') => {
                self.view = View::S3Config;
                self.s3_config_field = 0;
                self.s3_config_editing = false;
                self.s3_config_message = None;
            }
            KeyCode::Up if self.selected_list_item > 0 => {
                self.selected_list_item -= 1;
                self.list_state.select(Some(self.selected_list_item));
            }
            KeyCode::Down if self.selected_list_item < entry_count.saturating_sub(1) => {
                self.selected_list_item += 1;
                self.list_state.select(Some(self.selected_list_item));
            }
            KeyCode::Tab => {
                if let Some(ListEntry::Project { name }) = entries.get(self.selected_list_item) {
                    if self.expanded_projects.contains(name) {
                        self.expanded_projects.remove(name);
                    } else {
                        self.expanded_projects.insert(name.clone());
                    }
                }
            }
            KeyCode::Enter => match entries.get(self.selected_list_item) {
                Some(ListEntry::Project { name }) => {
                    if self.expanded_projects.contains(name) {
                        self.expanded_projects.remove(name);
                    } else {
                        self.expanded_projects.insert(name.clone());
                    }
                }
                Some(ListEntry::Run { run_index }) => {
                    self.selected_run = *run_index;
                    self.selected_card = 0;
                    self.view = View::RunDetail;
                    self.ensure_run_loaded(self.selected_run);
                }
                None => {}
            },
            KeyCode::Char(' ') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    if let Some(pos) = self.compared_runs.iter().position(|&i| i == *run_index) {
                        self.compared_runs.remove(pos);
                    } else {
                        self.compared_runs.push(*run_index);
                    }
                }
            }
            KeyCode::Char('v') => {
                if self.compared_runs.len() >= 2 {
                    self.view = View::Compare;
                    self.selected_card = 0;
                    self.scroll_offset = 0;
                    for idx in self.compared_runs.clone() {
                        self.ensure_run_loaded(idx);
                    }
                }
            }
            KeyCode::Char('p') => match entries.get(self.selected_list_item) {
                Some(ListEntry::Project { name }) => {
                    self.start_s3_pull_project(name);
                }
                Some(ListEntry::Run { run_index }) => {
                    self.selected_run = *run_index;
                    self.start_s3_pull();
                }
                None => {}
            },
            KeyCode::Char('P') => {
                self.pull_run_modal_open = true;
                self.pull_run_input.clear();
            }
            KeyCode::Char('s') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    let name = self.runs[*run_index].display_name();
                    if !self.starred_runs.remove(&name) {
                        self.starred_runs.insert(name);
                    }
                    save_starred_runs(&self.starred_runs);
                }
            }
            KeyCode::Char('n') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    let name = self.runs[*run_index].display_name();
                    self.note_modal_input = self.run_notes.get(&name).cloned().unwrap_or_default();
                    self.note_modal_run_name = name;
                    self.note_modal_open = true;
                }
            }
            KeyCode::Char('d') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    self.pending_delete_run = Some(*run_index);
                    self.show_delete_confirm = true;
                }
            }
            KeyCode::Char('M') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    let project = self.runs[*run_index].project.clone().unwrap_or_default();
                    self.move_run_input = project;
                    self.move_run_index = Some(*run_index);
                    self.move_run_modal_open = true;
                }
            }
            KeyCode::Char('f') => {
                self.filter_modal_open = true;
                self.filter_modal_phase = FilterPhase::KeySelect;
                self.filter_modal_input.clear();
                self.filter_modal_selected = 0;
                self.filter_modal_key.clear();
            }
            KeyCode::Char('F') => {
                self.config_filters.clear();
                self.selected_list_item = 0;
                self.list_state.select(Some(0));
            }
            KeyCode::Char('H') => {
                self.hide_completed = !self.hide_completed;
                self.selected_list_item = 0;
                self.list_state.select(Some(0));
            }
            KeyCode::Char('A') => {
                if let Some(ListEntry::Project { name }) = entries.get(self.selected_list_item) {
                    if !self.archived_projects.remove(name) {
                        self.archived_projects.insert(name.clone());
                    }
                    save_archived_projects(&self.archived_projects);
                    self.selected_list_item = 0;
                    self.list_state.select(Some(0));
                }
            }
            KeyCode::Char('.') => {
                self.show_archived = !self.show_archived;
                self.selected_list_item = 0;
                self.list_state.select(Some(0));
            }
            KeyCode::Char('/') => {
                self.search_editing = true;
            }
            KeyCode::Esc if !self.search_query.is_empty() => {
                self.search_query.clear();
                self.selected_list_item = 0;
                self.list_state.select(Some(0));
            }
            KeyCode::Esc if !self.compared_runs.is_empty() => {
                self.compared_runs.clear();
            }
            _ => {}
        }
    }

    fn handle_detail_key(&mut self, code: KeyCode, visible_rows: usize, cols: usize) {
        let card_count = self.card_count();
        let total_rows = card_count.div_ceil(cols);
        let max_scroll = total_rows.saturating_sub(visible_rows);

        let first_visible_row = self.scroll_offset;
        let last_visible_row = (self.scroll_offset + visible_rows).saturating_sub(1);

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.view = View::List,
            KeyCode::Left if self.selected_card > 0 => {
                self.selected_card -= 1;
                let new_row = self.selected_card / cols;
                if new_row < first_visible_row {
                    self.scroll_offset = new_row;
                }
            }
            KeyCode::Right if self.selected_card < card_count.saturating_sub(1) => {
                self.selected_card += 1;
                let new_row = self.selected_card / cols;
                if new_row > last_visible_row {
                    self.scroll_offset = (new_row + 1).saturating_sub(visible_rows).min(max_scroll);
                }
            }
            KeyCode::Enter if card_count > 0 => {
                self.selected_example = 0;
                self.cached_example = None;
                self.view = View::Focused;
            }
            KeyCode::Char('[') => {
                let current_project = self.runs[self.selected_run].project.clone();
                let prev = (0..self.selected_run).rev().find(|&i| {
                    self.runs[i].project == current_project
                        && (!self.hide_completed || self.runs[i].is_running())
                        && (self.config_filters.is_empty()
                            || run_matches_filters(&self.runs[i], &self.config_filters))
                });
                if let Some(idx) = prev {
                    self.selected_run = idx;
                    self.selected_card = 0;
                    self.scroll_offset = 0;
                    self.config_panel_scroll = 0;
                    self.cached_example = None;
                    self.ensure_run_loaded(self.selected_run);
                    self.sync_list_selection_to_run();
                }
            }
            KeyCode::Char(']') => {
                let current_project = self.runs[self.selected_run].project.clone();
                let next = (self.selected_run + 1..self.runs.len()).find(|&i| {
                    self.runs[i].project == current_project
                        && (!self.hide_completed || self.runs[i].is_running())
                        && (self.config_filters.is_empty()
                            || run_matches_filters(&self.runs[i], &self.config_filters))
                });
                if let Some(idx) = next {
                    self.selected_run = idx;
                    self.selected_card = 0;
                    self.scroll_offset = 0;
                    self.config_panel_scroll = 0;
                    self.cached_example = None;
                    self.ensure_run_loaded(self.selected_run);
                    self.sync_list_selection_to_run();
                }
            }
            KeyCode::Char('c') => {
                self.show_config = !self.show_config;
                self.config_panel_scroll = 0;
                self.ensure_card_visible();
            }
            KeyCode::Char('J') if self.show_config => {
                self.config_panel_scroll += 1;
            }
            KeyCode::Char('K') if self.show_config => {
                self.config_panel_scroll = self.config_panel_scroll.saturating_sub(1);
            }
            KeyCode::Char('C') if !self.runs.is_empty() => {
                if self.runs[self.selected_run].config.is_some() {
                    self.config_cursor = 0;
                    self.config_copied_at = None;
                    self.view = View::ConfigFull;
                }
            }
            KeyCode::Char('d') if !self.runs.is_empty() => {
                self.pending_delete_run = Some(self.selected_run);
                self.show_delete_confirm = true;
            }
            KeyCode::Char('p') => {
                self.start_s3_pull();
            }
            KeyCode::Char('s') if !self.runs.is_empty() => {
                let name = self.runs[self.selected_run].display_name();
                if !self.starred_runs.remove(&name) {
                    self.starred_runs.insert(name);
                }
                save_starred_runs(&self.starred_runs);
            }
            KeyCode::Char('n') if !self.runs.is_empty() => {
                let name = self.runs[self.selected_run].display_name();
                self.note_modal_input = self.run_notes.get(&name).cloned().unwrap_or_default();
                self.note_modal_run_name = name;
                self.note_modal_open = true;
            }
            KeyCode::Char('S') => {
                self.show_system_metrics = !self.show_system_metrics;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char('t') => {
                if let Some(Card::Chart { name }) = self.cards().get(self.selected_card) {
                    let name = name.clone();
                    self.toggle_chart_x_axis(&name);
                }
            }
            KeyCode::Char('m')
                if !self.runs.is_empty() && self.runs[self.selected_run].is_running() =>
            {
                self.show_complete_confirm = true;
            }
            KeyCode::Char('M') if !self.runs.is_empty() => {
                let project = self.runs[self.selected_run]
                    .project
                    .clone()
                    .unwrap_or_default();
                self.move_run_input = project;
                self.move_run_index = Some(self.selected_run);
                self.move_run_modal_open = true;
            }
            _ => {}
        }
    }

    fn handle_config_full_key(&mut self, code: KeyCode) {
        let line_count = self
            .current_run()
            .and_then(|r| r.config.as_ref())
            .map(|c| {
                let mut vals = Vec::new();
                flatten_config_values(c, &mut vals);
                vals.len()
            })
            .unwrap_or(0);

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.view = View::RunDetail,
            KeyCode::Up => self.config_cursor = self.config_cursor.saturating_sub(1),
            KeyCode::Down => {
                if line_count > 0 {
                    self.config_cursor = (self.config_cursor + 1).min(line_count - 1);
                }
            }
            KeyCode::PageUp => self.config_cursor = self.config_cursor.saturating_sub(20),
            KeyCode::PageDown => {
                if line_count > 0 {
                    self.config_cursor = (self.config_cursor + 20).min(line_count - 1);
                }
            }
            KeyCode::Char('y') | KeyCode::Enter => {
                if let Some(run) = self.current_run()
                    && let Some(config) = &run.config
                {
                    let mut vals = Vec::new();
                    flatten_config_values(config, &mut vals);
                    if let Some(val) = vals.get(self.config_cursor) {
                        copy_to_clipboard(val);
                        self.config_copied_at = Some(Instant::now());
                    }
                }
            }
            KeyCode::Char('Y') => {
                if let Some(run) = self.current_run()
                    && let Some(config) = &run.config
                    && let Ok(json) = serde_json::to_string_pretty(config)
                {
                    copy_to_clipboard(&json);
                    self.config_copied_at = Some(Instant::now());
                }
            }
            KeyCode::Char('[') => {
                let current_project = self.runs[self.selected_run].project.clone();
                let prev = (0..self.selected_run).rev().find(|&i| {
                    self.runs[i].project == current_project
                        && self.runs[i].config.is_some()
                        && (!self.hide_completed || self.runs[i].is_running())
                        && (self.config_filters.is_empty()
                            || run_matches_filters(&self.runs[i], &self.config_filters))
                });
                if let Some(idx) = prev {
                    self.selected_run = idx;
                    self.selected_card = 0;
                    self.scroll_offset = 0;
                    self.config_panel_scroll = 0;
                    self.config_cursor = 0;
                    self.config_copied_at = None;
                    self.cached_example = None;
                    self.ensure_run_loaded(self.selected_run);
                    self.sync_list_selection_to_run();
                }
            }
            KeyCode::Char(']') => {
                let current_project = self.runs[self.selected_run].project.clone();
                let next = (self.selected_run + 1..self.runs.len()).find(|&i| {
                    self.runs[i].project == current_project
                        && self.runs[i].config.is_some()
                        && (!self.hide_completed || self.runs[i].is_running())
                        && (self.config_filters.is_empty()
                            || run_matches_filters(&self.runs[i], &self.config_filters))
                });
                if let Some(idx) = next {
                    self.selected_run = idx;
                    self.selected_card = 0;
                    self.scroll_offset = 0;
                    self.config_panel_scroll = 0;
                    self.config_cursor = 0;
                    self.config_copied_at = None;
                    self.cached_example = None;
                    self.ensure_run_loaded(self.selected_run);
                    self.sync_list_selection_to_run();
                }
            }
            _ => {}
        }
    }

    fn handle_compare_key(&mut self, code: KeyCode, visible_rows: usize, cols: usize) {
        let cards = self.compare_cards();
        let card_count = cards.len();
        let total_rows = card_count.div_ceil(cols);
        let max_scroll = total_rows.saturating_sub(visible_rows);

        let first_visible_row = self.scroll_offset;
        let last_visible_row = (self.scroll_offset + visible_rows).saturating_sub(1);

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.view = View::List;
            }
            KeyCode::Left if self.selected_card > 0 => {
                self.selected_card -= 1;
                let new_row = self.selected_card / cols;
                if new_row < first_visible_row {
                    self.scroll_offset = new_row;
                }
            }
            KeyCode::Right if self.selected_card < card_count.saturating_sub(1) => {
                self.selected_card += 1;
                let new_row = self.selected_card / cols;
                if new_row > last_visible_row {
                    self.scroll_offset = (new_row + 1).saturating_sub(visible_rows).min(max_scroll);
                }
            }
            KeyCode::Enter if card_count > 0 => {
                self.compare_focused = true;
                self.view = View::Focused;
            }
            KeyCode::Char('p') => {
                self.start_s3_pull_compared();
            }
            KeyCode::Char('c') => {
                self.show_config = !self.show_config;
                self.config_panel_scroll = 0;
                self.ensure_card_visible();
            }
            KeyCode::Char('J') if self.show_config => {
                self.config_panel_scroll += 1;
            }
            KeyCode::Char('K') if self.show_config => {
                self.config_panel_scroll = self.config_panel_scroll.saturating_sub(1);
            }
            KeyCode::Char('S') => {
                self.show_system_metrics = !self.show_system_metrics;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char('t') => {
                if let Some(Card::Chart { name }) = self.compare_cards().get(self.selected_card) {
                    let name = name.clone();
                    self.toggle_chart_x_axis(&name);
                }
            }
            _ => {}
        }
    }

    fn handle_focused_key(&mut self, key: KeyEvent) {
        let code = key.code;
        let modifiers = key.modifiers;
        let (card_count, cards) = if self.compare_focused {
            let c = self.compare_cards();
            (c.len(), c)
        } else {
            match self.view_mode {
                ViewMode::Runs => (self.card_count(), self.cards()),
                ViewMode::Artifacts | ViewMode::Infra => (0, vec![]),
            }
        };
        let current_card = cards.get(self.selected_card);

        let checkpoint_count = match current_card {
            Some(Card::Checkpoints) => self.current_run().map(|r| r.checkpoints.len()).unwrap_or(0),
            _ => 0,
        };

        let example_count = match current_card {
            Some(Card::Examples { name }) if self.compare_focused => {
                self.compare_example_steps(name).len()
            }
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .map(|e| e.len())
                .unwrap_or(0),
            _ => 0,
        };

        let confusion_state = match current_card {
            Some(Card::ConfusionMatrix { name }) => self
                .current_run()
                .and_then(|r| r.confusion_matrices.get(name))
                .map(|s| (name.clone(), s.len())),
            _ => None,
        };

        let prompt_count = match current_card {
            Some(Card::Examples { name }) if self.compare_focused => {
                let steps = self.compare_example_steps(name);
                let target_step = steps.get(self.selected_example).copied();
                target_step
                    .map(|step| {
                        self.compared_runs
                            .iter()
                            .filter_map(|&idx| self.runs.get(idx))
                            .filter_map(|run| {
                                let g = run.examples.get(name)?;
                                let idx = g.find_step(step)?;
                                g.load(idx)
                            })
                            .map(|ex| ex.prompts.len())
                            .max()
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            }
            Some(Card::Examples { name }) => self
                .cached_example
                .as_ref()
                .filter(|(n, i, _)| n == name && *i == self.selected_example)
                .map(|(_, _, ex)| ex.prompts.len())
                .unwrap_or(0),
            _ => 0,
        };

        let response_count = match current_card {
            Some(Card::Examples { name }) if self.compare_focused => {
                let steps = self.compare_example_steps(name);
                let target_step = steps.get(self.selected_example).copied();
                target_step
                    .map(|step| {
                        self.compared_runs
                            .iter()
                            .filter_map(|&idx| self.runs.get(idx))
                            .filter_map(|run| {
                                let g = run.examples.get(name)?;
                                let idx = g.find_step(step)?;
                                g.load(idx)
                            })
                            .filter_map(|ex| ex.responses.get(self.selected_prompt).cloned())
                            .map(|r| r.len())
                            .max()
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            }
            Some(Card::Examples { name }) => self
                .cached_example
                .as_ref()
                .filter(|(n, i, _)| n == name && *i == self.selected_example)
                .and_then(|(_, _, ex)| ex.responses.get(self.selected_prompt))
                .map(|r| r.len())
                .unwrap_or(0),
            _ => 0,
        };

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.compare_focused {
                    self.view = View::Compare;
                    self.compare_focused = false;
                } else {
                    self.view = match self.view_mode {
                        ViewMode::Runs => View::RunDetail,
                        ViewMode::Artifacts => View::List,
                        ViewMode::Infra => View::InfraList,
                    };
                }
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::Char('t') => {
                if let Some(Card::Chart { name }) = current_card {
                    let name = name.clone();
                    self.toggle_chart_x_axis(&name);
                }
            }
            // Tab cycles focus: Prompt → Groundtruth → Response (skip Groundtruth when absent)
            KeyCode::Tab => {
                let has_gt = self.current_example_has_groundtruth();
                self.focused_section = match self.focused_section {
                    FocusedSection::Prompt => {
                        if has_gt {
                            FocusedSection::Groundtruth
                        } else {
                            FocusedSection::Response
                        }
                    }
                    FocusedSection::Groundtruth => FocusedSection::Response,
                    FocusedSection::Response => FocusedSection::Prompt,
                };
            }
            // k scrolls down, j scrolls up in the focused section
            KeyCode::Char('k') => {
                let max = self.focused_max_scroll(self.focused_section);
                match self.focused_section {
                    FocusedSection::Prompt => {
                        self.prompt_scroll_offset = (self.prompt_scroll_offset + 1).min(max);
                    }
                    FocusedSection::Groundtruth => {
                        self.groundtruth_scroll_offset =
                            (self.groundtruth_scroll_offset + 1).min(max);
                    }
                    FocusedSection::Response => {
                        self.response_scroll_offset = (self.response_scroll_offset + 1).min(max);
                    }
                }
            }
            KeyCode::Char('j') => match self.focused_section {
                FocusedSection::Prompt => {
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_sub(1);
                }
                FocusedSection::Groundtruth => {
                    self.groundtruth_scroll_offset =
                        self.groundtruth_scroll_offset.saturating_sub(1);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_sub(1);
                }
            },
            // PageDown/PageUp for faster scrolling
            KeyCode::PageDown => {
                let max = self.focused_max_scroll(self.focused_section);
                match self.focused_section {
                    FocusedSection::Prompt => {
                        self.prompt_scroll_offset = (self.prompt_scroll_offset + 10).min(max);
                    }
                    FocusedSection::Groundtruth => {
                        self.groundtruth_scroll_offset =
                            (self.groundtruth_scroll_offset + 10).min(max);
                    }
                    FocusedSection::Response => {
                        self.response_scroll_offset = (self.response_scroll_offset + 10).min(max);
                    }
                }
            }
            KeyCode::PageUp => match self.focused_section {
                FocusedSection::Prompt => {
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_sub(10);
                }
                FocusedSection::Groundtruth => {
                    self.groundtruth_scroll_offset =
                        self.groundtruth_scroll_offset.saturating_sub(10);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_sub(10);
                }
            },
            // Left/Right navigate between cards
            KeyCode::Left if card_count > 1 && self.selected_card > 0 => {
                self.selected_card -= 1;
                self.selected_example = 0;
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.selected_checkpoint = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::Right
                if card_count > 1 && self.selected_card < card_count.saturating_sub(1) =>
            {
                self.selected_card += 1;
                self.selected_example = 0;
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.selected_checkpoint = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            // Checkpoint navigation
            KeyCode::Up if checkpoint_count > 0 && self.selected_checkpoint > 0 => {
                self.selected_checkpoint -= 1;
            }
            KeyCode::Down
                if checkpoint_count > 0
                    && self.selected_checkpoint < checkpoint_count.saturating_sub(1) =>
            {
                self.selected_checkpoint += 1;
            }
            KeyCode::Char('p') if checkpoint_count > 0 => {
                if let Some(run) = self.current_run()
                    && let Some(ckpt) = run.checkpoints.get(self.selected_checkpoint)
                    && !ckpt.all_downloaded()
                {
                    let step = ckpt.step;
                    if ckpt.is_legacy() {
                        self.start_s3_pull_checkpoint(step, None);
                    } else {
                        let mut options = Vec::new();
                        for f in &ckpt.files {
                            if !ckpt.downloaded_files.contains(&f.name) {
                                options.push(f.name.clone());
                            }
                        }
                        if options.len() > 1 {
                            options.push("All files".to_string());
                        }
                        self.checkpoint_download_step = step;
                        self.checkpoint_download_options = options;
                        self.checkpoint_download_selected = 0;
                        self.checkpoint_download_modal = true;
                    }
                }
            }
            KeyCode::Char('D') if checkpoint_count > 0 => {
                self.pending_delete_checkpoint = true;
                self.show_delete_confirm = true;
            }
            // Confusion matrix step scrubbing
            KeyCode::Up if confusion_state.is_some() => {
                if let Some((name, len)) = confusion_state.clone() {
                    let cur = self
                        .confusion_step_idx
                        .get(&name)
                        .copied()
                        .unwrap_or_else(|| len.saturating_sub(1));
                    let step = if modifiers.contains(KeyModifiers::SHIFT) {
                        10
                    } else {
                        1
                    };
                    self.confusion_step_idx
                        .insert(name, cur.saturating_sub(step));
                }
            }
            KeyCode::Down if confusion_state.is_some() => {
                if let Some((name, len)) = confusion_state.clone() {
                    let cur = self
                        .confusion_step_idx
                        .get(&name)
                        .copied()
                        .unwrap_or_else(|| len.saturating_sub(1));
                    let step = if modifiers.contains(KeyModifiers::SHIFT) {
                        10
                    } else {
                        1
                    };
                    let next = (cur + step).min(len.saturating_sub(1));
                    self.confusion_step_idx.insert(name, next);
                }
            }
            KeyCode::Home if confusion_state.is_some() => {
                if let Some((name, _)) = confusion_state.clone() {
                    self.confusion_step_idx.insert(name, 0);
                }
            }
            KeyCode::End if confusion_state.is_some() => {
                if let Some((name, len)) = confusion_state.clone() {
                    self.confusion_step_idx.insert(name, len.saturating_sub(1));
                }
            }
            // Shift+Up/Down jump 10 examples at a time
            KeyCode::Up if example_count > 0 && modifiers.contains(KeyModifiers::SHIFT) => {
                self.selected_example = self.selected_example.saturating_sub(10);
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::Down if example_count > 0 && modifiers.contains(KeyModifiers::SHIFT) => {
                self.selected_example =
                    (self.selected_example + 10).min(example_count.saturating_sub(1));
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            // Up/Down navigate within example groups
            KeyCode::Up if example_count > 0 && self.selected_example > 0 => {
                self.selected_example -= 1;
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::Down
                if example_count > 0 && self.selected_example < example_count.saturating_sub(1) =>
            {
                self.selected_example += 1;
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            // [ and ] navigate between prompts in batch
            KeyCode::Char('[') if prompt_count > 1 && self.selected_prompt > 0 => {
                self.selected_prompt -= 1;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::Char(']')
                if prompt_count > 1 && self.selected_prompt < prompt_count.saturating_sub(1) =>
            {
                self.selected_prompt += 1;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            // < and > navigate between response variants
            KeyCode::Char('<') if response_count > 1 && self.selected_response > 0 => {
                self.selected_response -= 1;
                self.response_scroll_offset = 0;
            }
            KeyCode::Char('>')
                if response_count > 1
                    && self.selected_response < response_count.saturating_sub(1) =>
            {
                self.selected_response += 1;
                self.response_scroll_offset = 0;
            }
            // Home/End jump to first/last example
            KeyCode::Home if example_count > 0 => {
                self.selected_example = 0;
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            KeyCode::End if example_count > 0 => {
                self.selected_example = example_count.saturating_sub(1);
                self.cached_example = None;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
                self.groundtruth_scroll_offset = 0;
            }
            // g opens goto step input
            KeyCode::Char('g') if example_count > 0 => {
                self.goto_step_input = Some(String::new());
            }
            KeyCode::Char('J') if self.show_config => {
                self.config_panel_scroll += 1;
            }
            KeyCode::Char('K') if self.show_config => {
                self.config_panel_scroll = self.config_panel_scroll.saturating_sub(1);
            }
            _ => {}
        }
    }

    fn handle_terminate_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some(idx) = self.pending_terminate_instance {
                    self.terminate_instance(idx);
                }
                self.show_terminate_confirm = false;
                self.pending_terminate_instance = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.show_terminate_confirm = false;
                self.pending_terminate_instance = None;
            }
            _ => {}
        }
    }

    fn handle_complete_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                let run = &mut self.runs[self.selected_run];
                if mark_run_completed(&run.path).is_ok() {
                    run.status = data::RunStatus::Completed;
                    run.end_time = Some(chrono::Local::now());
                }
                self.show_complete_confirm = false;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.show_complete_confirm = false;
            }
            _ => {}
        }
    }

    fn handle_artifacts_list_key(&mut self, code: KeyCode) {
        let count = self.artifacts.len();

        match code {
            KeyCode::Char('r') => {
                self.view_mode = ViewMode::Runs;
            }
            KeyCode::Char('i') => {
                self.view_mode = ViewMode::Infra;
                self.view = View::InfraList;
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Up if self.selected_artifact > 0 => {
                self.selected_artifact -= 1;
            }
            KeyCode::Down if self.selected_artifact < count.saturating_sub(1) => {
                self.selected_artifact += 1;
            }
            KeyCode::Enter if count > 0 => {
                self.artifact_detail_scroll = 0;
                self.view = View::ArtifactDetail;
            }
            KeyCode::Char('p') => {
                self.start_s3_pull_artifacts();
            }
            KeyCode::Char('d') if count > 0 => {
                self.start_s3_download_artifact();
            }
            KeyCode::Char('D') if count > 0 => {
                self.pending_delete_artifact = true;
                self.show_delete_confirm = true;
            }
            _ => {}
        }
    }

    fn handle_artifact_detail_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.view = View::List;
            }
            KeyCode::Up => {
                self.artifact_detail_scroll = self.artifact_detail_scroll.saturating_sub(1);
            }
            KeyCode::Down => {
                self.artifact_detail_scroll += 1;
            }
            KeyCode::Char('d') => {
                self.start_s3_download_artifact();
            }
            KeyCode::Char('D') => {
                self.pending_delete_artifact = true;
                self.show_delete_confirm = true;
            }
            KeyCode::Char('g') => {
                if let Some(artifact) = self.artifacts.get(self.selected_artifact)
                    && let (Some(proj), Some(rn)) = (&artifact.run_project, &artifact.run_name)
                {
                    let target_display = format!("{}/{}", proj, rn);
                    if let Some(idx) = self
                        .runs
                        .iter()
                        .position(|r| r.display_name() == target_display)
                    {
                        self.selected_run = idx;
                        self.ensure_run_loaded(idx);
                        self.view_mode = ViewMode::Runs;
                        self.view = View::RunDetail;
                        self.scroll_offset = 0;
                        self.selected_card = 0;
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_infra_list_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if self.add_machine_open {
            self.handle_add_machine_key(code);
            return;
        }
        if self.launch_confirming {
            self.handle_infra_launch_confirm_key(code);
            return;
        }
        if self.launch_selecting_region {
            self.handle_infra_region_select_key(code);
            return;
        }

        let instances = self.filtered_infra_instances();
        let instance_count = instances.len();
        let type_count = self.infra_types.len();

        match code {
            KeyCode::Char('r') => {
                self.view_mode = ViewMode::Runs;
                self.view = View::List;
            }
            KeyCode::Char('a')
                if !(self.infra_active_panel == InfraPanel::Instances
                    && self.selected_infra_provider == Provider::Local) =>
            {
                self.view_mode = ViewMode::Artifacts;
                self.view = View::List;
            }
            KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
                self.infra_active_panel = match self.infra_active_panel {
                    InfraPanel::Instances => InfraPanel::Types,
                    InfraPanel::Types => InfraPanel::Instances,
                };
            }
            KeyCode::Up => match self.infra_active_panel {
                InfraPanel::Instances if self.selected_infra_instance > 0 => {
                    self.selected_infra_instance -= 1;
                }
                InfraPanel::Types if self.selected_infra_type > 0 => {
                    self.selected_infra_type -= 1;
                    self.infra_types_list_state
                        .select(Some(self.selected_infra_type));
                }
                _ => {}
            },
            KeyCode::Down => match self.infra_active_panel {
                InfraPanel::Instances
                    if self.selected_infra_instance < instance_count.saturating_sub(1) =>
                {
                    self.selected_infra_instance += 1;
                }
                InfraPanel::Types if self.selected_infra_type < type_count.saturating_sub(1) => {
                    self.selected_infra_type += 1;
                    self.infra_types_list_state
                        .select(Some(self.selected_infra_type));
                }
                _ => {}
            },
            KeyCode::Char('1') => {
                self.selected_infra_provider = Provider::Vast;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.infra_types_list_state = ListState::default();
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('2') => {
                self.selected_infra_provider = Provider::Prime;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.infra_types_list_state = ListState::default();
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('3') => {
                self.selected_infra_provider = Provider::Lambda;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.infra_types_list_state = ListState::default();
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('4') => {
                self.selected_infra_provider = Provider::Local;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.infra_types_list_state = ListState::default();
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Enter => match self.infra_active_panel {
                InfraPanel::Instances => {
                    let instances = self.filtered_infra_instances();
                    if let Some(instance) = instances.get(self.selected_infra_instance)
                        && instance.ip.is_some()
                    {
                        let instance_clone = (*instance).clone();
                        if modifiers.contains(KeyModifiers::SHIFT) {
                            self.open_session_modal(&instance_clone);
                        } else {
                            let _ = self.launch_ssh(&instance_clone);
                        }
                    }
                }
                InfraPanel::Types if type_count > 0 => {
                    if let Some(instance_type) = self.infra_types.get(self.selected_infra_type) {
                        if instance_type.regions.is_empty() {
                            self.infra_error = Some("No availability for this type".to_string());
                        } else if instance_type.regions.len() == 1 {
                            self.launch_selected_region = 0;
                            self.launch_name_input.clear();
                            self.launch_confirming = true;
                        } else {
                            self.launch_selected_region = 0;
                            self.launch_selecting_region = true;
                        }
                    }
                }
                _ => {}
            },
            KeyCode::Char('S') if self.infra_active_panel == InfraPanel::Instances => {
                let instances = self.filtered_infra_instances();
                if let Some(instance) = instances.get(self.selected_infra_instance)
                    && instance.ip.is_some()
                {
                    let instance_clone = (*instance).clone();
                    self.open_session_modal(&instance_clone);
                }
            }
            KeyCode::Char('x') if self.infra_active_panel == InfraPanel::Instances => {
                if instance_count > 0 && self.selected_infra_instance < instance_count {
                    if self.selected_infra_provider == Provider::Local {
                        self.remove_local_machine(self.selected_infra_instance);
                    } else {
                        self.pending_terminate_instance = Some(self.selected_infra_instance);
                        self.show_terminate_confirm = true;
                    }
                }
            }
            KeyCode::Char('a')
                if self.infra_active_panel == InfraPanel::Instances
                    && self.selected_infra_provider == Provider::Local =>
            {
                self.add_machine_open = true;
                self.add_machine_name.clear();
                self.add_machine_user.clear();
                self.add_machine_host.clear();
                self.add_machine_focus = AddMachineField::User;
            }
            KeyCode::Char('R') => {
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('c') => {
                self.view = View::InfraConfig;
                self.config_provider_index = 0;
                self.config_editing_key = false;
                self.config_api_key_input.clear();
            }
            KeyCode::Char('s') if self.infra_active_panel == InfraPanel::Types => {
                self.infra_type_sort = self.infra_type_sort.next();
                self.sort_infra_types();
            }
            _ => {}
        }
    }

    fn handle_infra_launch_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.do_launch_instance();
                self.launch_confirming = false;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.launch_confirming = false;
            }
            _ => {}
        }
    }

    fn handle_infra_region_select_key(&mut self, code: KeyCode) {
        let regions_count = self
            .infra_types
            .get(self.selected_infra_type)
            .map(|t| t.regions.len())
            .unwrap_or(0);

        match code {
            KeyCode::Esc => {
                self.launch_selecting_region = false;
            }
            KeyCode::Up if self.launch_selected_region > 0 => {
                self.launch_selected_region -= 1;
            }
            KeyCode::Down if self.launch_selected_region < regions_count.saturating_sub(1) => {
                self.launch_selected_region += 1;
            }
            KeyCode::Enter => {
                self.launch_selecting_region = false;
                self.launch_confirming = true;
            }
            _ => {}
        }
    }

    fn handle_infra_config_key(&mut self, code: KeyCode) {
        let providers = Provider::all();

        if self.config_editing_key {
            match code {
                KeyCode::Esc => {
                    self.config_editing_key = false;
                    self.config_api_key_input.clear();
                }
                KeyCode::Enter => {
                    let provider = providers[self.config_provider_index];
                    let api_key = if self.config_api_key_input.is_empty() {
                        None
                    } else {
                        Some(self.config_api_key_input.clone())
                    };

                    match provider {
                        Provider::Lambda => self.infra_config.lambda_config.api_key = api_key,
                        Provider::Vast => self.infra_config.vast.api_key = api_key,
                        Provider::Prime => self.infra_config.prime.api_key = api_key,
                        Provider::Local => {}
                    }

                    let _ = save_config(&self.infra_config);
                    self.config_editing_key = false;
                    self.config_api_key_input.clear();
                }
                KeyCode::Backspace => {
                    self.config_api_key_input.pop();
                }
                KeyCode::Char(c) => {
                    self.config_api_key_input.push(c);
                }
                _ => {}
            }
        } else {
            match code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.view = View::InfraList;
                }
                KeyCode::Up if self.config_provider_index > 0 => {
                    self.config_provider_index -= 1;
                }
                KeyCode::Down if self.config_provider_index < providers.len().saturating_sub(1) => {
                    self.config_provider_index += 1;
                }
                KeyCode::Enter | KeyCode::Char('e') => {
                    let provider = providers[self.config_provider_index];
                    if provider == Provider::Local {
                        return;
                    }
                    self.config_editing_key = true;
                    let current_key = match provider {
                        Provider::Lambda => &self.infra_config.lambda_config.api_key,
                        Provider::Vast => &self.infra_config.vast.api_key,
                        Provider::Prime => &self.infra_config.prime.api_key,
                        Provider::Local => unreachable!(),
                    };
                    self.config_api_key_input = current_key.clone().unwrap_or_default();
                }
                KeyCode::Char('d') => {
                    let provider = providers[self.config_provider_index];
                    match provider {
                        Provider::Lambda => self.infra_config.lambda_config.api_key = None,
                        Provider::Vast => self.infra_config.vast.api_key = None,
                        Provider::Prime => self.infra_config.prime.api_key = None,
                        Provider::Local => {}
                    }
                    let _ = save_config(&self.infra_config);
                }
                _ => {}
            }
        }
    }

    fn handle_add_machine_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.add_machine_open = false;
            }
            KeyCode::Tab => {
                self.add_machine_focus = match self.add_machine_focus {
                    AddMachineField::User => AddMachineField::Host,
                    AddMachineField::Host => AddMachineField::Name,
                    AddMachineField::Name => AddMachineField::User,
                };
            }
            KeyCode::BackTab => {
                self.add_machine_focus = match self.add_machine_focus {
                    AddMachineField::User => AddMachineField::Name,
                    AddMachineField::Host => AddMachineField::User,
                    AddMachineField::Name => AddMachineField::Host,
                };
            }
            KeyCode::Enter => {
                if !self.add_machine_user.is_empty() && !self.add_machine_host.is_empty() {
                    let machine = LocalMachine {
                        name: if self.add_machine_name.is_empty() {
                            None
                        } else {
                            Some(self.add_machine_name.clone())
                        },
                        ssh_user: self.add_machine_user.clone(),
                        host: self.add_machine_host.clone(),
                    };
                    self.infra_config.local.push(machine);
                    let _ = save_config(&self.infra_config);
                    self.add_machine_open = false;
                    self.refresh_infra();
                }
            }
            KeyCode::Backspace => match self.add_machine_focus {
                AddMachineField::User => {
                    self.add_machine_user.pop();
                }
                AddMachineField::Host => {
                    self.add_machine_host.pop();
                }
                AddMachineField::Name => {
                    self.add_machine_name.pop();
                }
            },
            KeyCode::Char(c) => match self.add_machine_focus {
                AddMachineField::User => self.add_machine_user.push(c),
                AddMachineField::Host => self.add_machine_host.push(c),
                AddMachineField::Name => self.add_machine_name.push(c),
            },
            _ => {}
        }
    }

    fn remove_local_machine(&mut self, index: usize) {
        if index < self.infra_config.local.len() {
            self.infra_config.local.remove(index);
            let _ = save_config(&self.infra_config);
            self.refresh_infra();
        }
    }

    fn handle_s3_config_key(&mut self, code: KeyCode) {
        const FIELD_COUNT: usize = 6;
        let field_names = [
            "bucket",
            "prefix",
            "region",
            "access_key_id",
            "secret_access_key",
            "endpoint_url",
        ];

        if self.s3_config_editing {
            match code {
                KeyCode::Esc => {
                    self.s3_config_editing = false;
                    self.s3_config_input.clear();
                }
                KeyCode::Enter => {
                    let value = if self.s3_config_input.is_empty() {
                        None
                    } else {
                        Some(self.s3_config_input.clone())
                    };

                    match self.s3_config_field {
                        0 => self.s3_config.bucket = value.unwrap_or_default(),
                        1 => self.s3_config.prefix = value.unwrap_or_default(),
                        2 => self.s3_config.region = value,
                        3 => self.s3_config.access_key_id = value,
                        4 => self.s3_config.secret_access_key = value,
                        5 => self.s3_config.endpoint_url = value,
                        _ => {}
                    }

                    match s3::save_config(&self.s3_config) {
                        Ok(()) => self.s3_config_message = Some("Saved".to_string()),
                        Err(e) => self.s3_config_message = Some(format!("Error: {}", e)),
                    }
                    self.s3_config_editing = false;
                    self.s3_config_input.clear();
                }
                KeyCode::Backspace => {
                    self.s3_config_input.pop();
                }
                KeyCode::Char(c) => {
                    self.s3_config_input.push(c);
                }
                _ => {}
            }
        } else {
            match code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.view = View::List;
                    self.view_mode = ViewMode::Runs;
                }
                KeyCode::Up if self.s3_config_field > 0 => {
                    self.s3_config_field -= 1;
                    self.s3_config_message = None;
                }
                KeyCode::Down if self.s3_config_field < FIELD_COUNT - 1 => {
                    self.s3_config_field += 1;
                    self.s3_config_message = None;
                }
                KeyCode::Enter | KeyCode::Char('e') => {
                    self.s3_config_editing = true;
                    self.s3_config_input = match self.s3_config_field {
                        0 => self.s3_config.bucket.clone(),
                        1 => self.s3_config.prefix.clone(),
                        2 => self.s3_config.region.clone().unwrap_or_default(),
                        3 => self.s3_config.access_key_id.clone().unwrap_or_default(),
                        4 => self.s3_config.secret_access_key.clone().unwrap_or_default(),
                        5 => self.s3_config.endpoint_url.clone().unwrap_or_default(),
                        _ => String::new(),
                    };
                    self.s3_config_message = None;
                }
                KeyCode::Char('d') => {
                    match self.s3_config_field {
                        0 => self.s3_config.bucket.clear(),
                        1 => self.s3_config.prefix.clear(),
                        2 => self.s3_config.region = None,
                        3 => self.s3_config.access_key_id = None,
                        4 => self.s3_config.secret_access_key = None,
                        5 => self.s3_config.endpoint_url = None,
                        _ => {}
                    }
                    match s3::save_config(&self.s3_config) {
                        Ok(()) => self.s3_config_message = Some("Cleared".to_string()),
                        Err(e) => self.s3_config_message = Some(format!("Error: {}", e)),
                    }
                }
                _ => {}
            }
        }
        let _ = field_names;
    }

    fn handle_session_modal_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.session_modal_open = false;
                self.session_modal_instance = None;
            }
            KeyCode::Tab => {
                self.session_modal_focus = match self.session_modal_focus {
                    SessionModalField::PythonVersion => SessionModalField::RepoPath,
                    SessionModalField::RepoPath => SessionModalField::Command,
                    SessionModalField::Command => SessionModalField::SkipTmux,
                    SessionModalField::SkipTmux => SessionModalField::PythonVersion,
                };
            }
            KeyCode::BackTab => {
                self.session_modal_focus = match self.session_modal_focus {
                    SessionModalField::PythonVersion => SessionModalField::SkipTmux,
                    SessionModalField::RepoPath => SessionModalField::PythonVersion,
                    SessionModalField::Command => SessionModalField::RepoPath,
                    SessionModalField::SkipTmux => SessionModalField::Command,
                };
            }
            KeyCode::Char(' ') if self.session_modal_focus == SessionModalField::SkipTmux => {
                self.session_skip_tmux = !self.session_skip_tmux;
            }
            KeyCode::Char(c) => match self.session_modal_focus {
                SessionModalField::PythonVersion => self.session_python_version.push(c),
                SessionModalField::RepoPath => self.session_repo_path.push(c),
                SessionModalField::Command => self.session_command.push(c),
                SessionModalField::SkipTmux => {}
            },
            KeyCode::Backspace => match self.session_modal_focus {
                SessionModalField::PythonVersion => {
                    self.session_python_version.pop();
                }
                SessionModalField::RepoPath => {
                    self.session_repo_path.pop();
                }
                SessionModalField::Command => {
                    self.session_command.pop();
                }
                SessionModalField::SkipTmux => {}
            },
            KeyCode::Enter => {
                if let Some(instance) = self.session_modal_instance.take() {
                    self.do_launch_ssh_with_setup(&instance);
                }
                self.session_modal_open = false;
            }
            _ => {}
        }
    }

    fn open_session_modal(&mut self, instance: &Instance) {
        self.session_skip_tmux = false;
        self.session_modal_instance = Some(instance.clone());
        self.session_modal_focus = SessionModalField::PythonVersion;
        self.session_modal_open = true;
    }

    fn do_launch_instance(&mut self) {
        let provider = self.selected_infra_provider;
        let config = self.infra_config.get_provider_config(provider);

        let Some(api_key) = &config.api_key else {
            self.infra_error = Some(format!("No API key for {}", provider.display_name()));
            return;
        };

        let Some(instance_type) = self.infra_types.get(self.selected_infra_type) else {
            self.infra_error = Some("No instance type selected".to_string());
            return;
        };

        let region = if instance_type.regions.is_empty() {
            config.default_region.clone()
        } else {
            instance_type
                .regions
                .get(self.launch_selected_region)
                .cloned()
        };

        let name = if self.launch_name_input.is_empty() {
            None
        } else {
            Some(self.launch_name_input.clone())
        };

        let ssh_key_names = config.ssh_key_name.as_ref().map(|k| vec![k.clone()]);

        let opts = infra::models::LaunchOptions {
            instance_type: instance_type.name.clone(),
            region,
            ssh_key_names,
            name,
            metadata: instance_type.metadata.clone(),
        };

        let client = get_provider(provider, api_key);
        match client.launch(&opts) {
            Ok(ids) => {
                self.infra_error = None;
                self.infra_active_panel = InfraPanel::Instances;
                self.infra_last_refresh = Instant::now() - Duration::from_secs(2);
                if !ids.is_empty() {
                    self.infra_error = Some(format!("Launched: {}", ids.join(", ")));
                    self.infra_message_time = Some(Instant::now());
                }
            }
            Err(e) => {
                self.infra_error = Some(format!("Launch failed: {}", e));
            }
        }
    }
}

fn main() -> Result<()> {
    let command = parse_command()?;

    match command {
        Command::Tui(options) => run_tui(options),
        Command::Pull(options) => run_s3_command("pull", options),
        Command::Push(options) => run_s3_command("push", options),
        Command::Sync(options) => run_s3_command("sync", options),
        Command::Run(options) => run::run(options),
        Command::PruneLocal(options) => prune::run_local(options),
    }
}

fn run_tui(_options: TuiOptions) -> Result<()> {
    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run event loop
    let mut app = App::new();
    let mut last_list_refresh = Instant::now();
    let list_refresh_interval = Duration::from_secs(3);

    while !app.should_quit {
        // Update terminal size
        let size = terminal.size()?;
        app.update_size(size.width, size.height);

        // Periodic refresh of run list (less frequent)
        if last_list_refresh.elapsed() >= list_refresh_interval {
            app.refresh_runs();
            if matches!(app.view, View::RunDetail | View::Focused) && !app.compare_focused {
                app.ensure_run_loaded(app.selected_run);
            }
            if app.view == View::Compare || (app.view == View::Focused && app.compare_focused) {
                for idx in app.compared_runs.clone() {
                    app.ensure_run_loaded(idx);
                }
            }
            last_list_refresh = Instant::now();
        }

        // Auto-refresh infra instances when viewing infra tab (every 30 seconds)
        if app.view == View::InfraList
            && app.selected_infra_provider != Provider::Local
            && app.infra_last_refresh.elapsed() >= Duration::from_secs(30)
        {
            app.refresh_infra();
        }

        // Poll setup background task
        if let Some(rx) = &app.setup_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    SetupMessage::Status(s) => {
                        app.infra_error = Some(s);
                        app.infra_message_time = Some(Instant::now());
                    }
                    SetupMessage::Done(s) => {
                        app.infra_error = Some(s);
                        app.infra_message_time = Some(Instant::now());
                        app.setup_rx = None;
                        break;
                    }
                    SetupMessage::Error(s) => {
                        app.infra_error = Some(s);
                        app.infra_message_time = Some(Instant::now());
                        app.setup_rx = None;
                        break;
                    }
                }
            }
        }

        // Clear transient infra messages after 3 seconds
        if let Some(msg_time) = app.infra_message_time
            && msg_time.elapsed() >= Duration::from_secs(3)
            && app.setup_rx.is_none()
        {
            app.infra_error = None;
            app.infra_message_time = None;
        }

        // Poll S3 pull background task
        if let Some(rx) = &app.s3_pull_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    S3PullMessage::Pulling(s) => {
                        app.s3_pull_status = Some(s);
                    }
                    S3PullMessage::Done(s) => {
                        app.s3_pull_status = Some(s);
                        app.s3_pull_time = Some(Instant::now());
                        app.s3_pull_rx = None;
                        app.refresh_runs();
                        app.refresh_artifacts();
                        if let Some(run) = app.runs.get(app.selected_run) {
                            let path = run.path.clone();
                            if let Some(updated) = data::reload_run(&path) {
                                app.runs[app.selected_run] = updated;
                            }
                        }
                        if app.view == View::Compare
                            || (app.view == View::Focused && app.compare_focused)
                        {
                            for idx in app.compared_runs.clone() {
                                app.ensure_run_loaded(idx);
                            }
                        }
                        break;
                    }
                    S3PullMessage::Error(s) => {
                        app.s3_pull_status = Some(s);
                        app.s3_pull_time = Some(Instant::now());
                        app.s3_pull_rx = None;
                        break;
                    }
                }
            }
        }

        // Clear S3 pull status after 3 seconds
        if let Some(pull_time) = app.s3_pull_time
            && pull_time.elapsed() >= Duration::from_secs(3)
            && app.s3_pull_rx.is_none()
        {
            app.s3_pull_status = None;
            app.s3_pull_time = None;
        }

        if app.view == View::Focused && !app.compare_focused {
            let cards = app.active_cards();
            if let Some(Card::Examples { name }) = cards.get(app.selected_card) {
                let name = name.clone();
                let idx = app.selected_example;
                app.get_or_load_example(&name, idx);
            }
        }

        // Draw the UI
        terminal.draw(|frame| render(&mut app, frame))?;

        // Handle input (with 100ms timeout for responsive feel)
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            // Only handle key press, not release
            if key.kind == KeyEventKind::Press {
                app.handle_key(key);
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn run_s3_command(cmd: &str, options: SyncOptions) -> Result<()> {
    let config = s3::load_config()?.ok_or_else(|| {
        anyhow::anyhow!(
            "S3 not configured. Create ~/.extty/s3/config.toml with:\n\n\
             bucket = \"your-bucket\"\n\
             region = \"us-west-2\"\n"
        )
    })?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let client = s3::S3Client::new(config).await?;
        let runs_dir = runs_dir();

        let (project_filter, run_filter) = parse_target(&options.target);

        match cmd {
            "pull" => {
                let runs = client.list_runs(project_filter.as_deref()).await?;
                let runs_to_sync: Vec<_> = runs
                    .into_iter()
                    .filter(|r| run_filter.as_ref().is_none_or(|rf| r.name == *rf))
                    .collect();

                if runs_to_sync.is_empty() {
                    println!("No runs found to pull.");
                    return Ok(());
                }

                for run in runs_to_sync {
                    client
                        .download_run(
                            &run.project,
                            &run.name,
                            &runs_dir,
                            options.force,
                            options.dry_run,
                        )
                        .await?;
                    if !options.dry_run {
                        println!("Pulled: {}/{}", run.project, run.name);
                    }
                }
            }
            "push" => {
                let local_runs = list_local_runs(&runs_dir, project_filter.as_deref())?;
                let runs_to_sync: Vec<_> = local_runs
                    .into_iter()
                    .filter(|(_, name)| run_filter.as_ref().is_none_or(|rf| name == rf))
                    .collect();

                if runs_to_sync.is_empty() {
                    println!("No runs found to push.");
                    return Ok(());
                }

                for (project, name) in runs_to_sync {
                    client
                        .upload_run(&project, &name, &runs_dir, options.force, options.dry_run)
                        .await?;
                    if !options.dry_run {
                        println!("Pushed: {}/{}", project, name);
                    }
                }
            }
            "sync" => {
                let remote_runs = client.list_runs(project_filter.as_deref()).await?;
                let local_runs = list_local_runs(&runs_dir, project_filter.as_deref())?;

                let mut all_runs: std::collections::HashSet<(String, String)> =
                    std::collections::HashSet::new();
                for r in &remote_runs {
                    all_runs.insert((r.project.clone(), r.name.clone()));
                }
                for (p, n) in &local_runs {
                    all_runs.insert((p.clone(), n.clone()));
                }

                let runs_to_sync: Vec<_> = all_runs
                    .into_iter()
                    .filter(|(_, name)| run_filter.as_ref().is_none_or(|rf| name == rf))
                    .collect();

                if runs_to_sync.is_empty() {
                    println!("No runs found to sync.");
                    return Ok(());
                }

                for (project, name) in runs_to_sync {
                    client
                        .sync_run(&project, &name, &runs_dir, options.dry_run)
                        .await?;
                    if !options.dry_run {
                        println!("Synced: {}/{}", project, name);
                    }
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    })
}

fn parse_target(target: &Option<String>) -> (Option<String>, Option<String>) {
    match target {
        None => (None, None),
        Some(t) => {
            if t.ends_with('/') {
                (Some(t.trim_end_matches('/').to_string()), None)
            } else if t.contains('/') {
                let parts: Vec<&str> = t.splitn(2, '/').collect();
                (Some(parts[0].to_string()), Some(parts[1].to_string()))
            } else {
                (Some(t.clone()), None)
            }
        }
    }
}

fn runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("runs")
}

fn list_local_runs(
    runs_dir: &PathBuf,
    project_filter: Option<&str>,
) -> Result<Vec<(String, String)>> {
    let mut runs = Vec::new();

    if !runs_dir.exists() {
        return Ok(runs);
    }

    for proj_entry in std::fs::read_dir(runs_dir)? {
        let proj_entry = proj_entry?;
        let proj_path = proj_entry.path();
        if !proj_path.is_dir() {
            continue;
        }

        let proj_name = proj_entry.file_name().to_string_lossy().to_string();
        if let Some(filter) = project_filter
            && proj_name != filter
        {
            continue;
        }

        for run_entry in std::fs::read_dir(&proj_path)? {
            let run_entry = run_entry?;
            let run_path = run_entry.path();
            if run_path.is_dir() && run_path.join("meta.json").exists() {
                runs.push((
                    proj_name.clone(),
                    run_entry.file_name().to_string_lossy().to_string(),
                ));
            }
        }
    }

    Ok(runs)
}

enum Command {
    Tui(TuiOptions),
    Pull(SyncOptions),
    Push(SyncOptions),
    Sync(SyncOptions),
    Run(run::RunOptions),
    PruneLocal(prune::PruneLocalOptions),
}

struct TuiOptions;

struct SyncOptions {
    target: Option<String>,
    force: bool,
    dry_run: bool,
}

fn parse_command() -> Result<Command> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Ok(Command::Tui(TuiOptions));
    }

    match args[0].as_str() {
        "pull" | "push" | "sync" => {
            let cmd = args.remove(0);
            let opts = parse_sync_options(&mut args)?;
            match cmd.as_str() {
                "pull" => Ok(Command::Pull(opts)),
                "push" => Ok(Command::Push(opts)),
                "sync" => Ok(Command::Sync(opts)),
                _ => unreachable!(),
            }
        }
        "run" => {
            args.remove(0);
            let opts = parse_run_options(&mut args)?;
            Ok(Command::Run(opts))
        }
        "prune" => {
            args.remove(0);
            let sub = args
                .first()
                .ok_or_else(|| anyhow::anyhow!("prune requires a subcommand: local"))?
                .clone();
            args.remove(0);
            match sub.as_str() {
                "local" => {
                    let opts = parse_prune_local_options(&mut args)?;
                    Ok(Command::PruneLocal(opts))
                }
                other => Err(anyhow::anyhow!(
                    "Unknown prune subcommand: {}. Valid: local",
                    other
                )),
            }
        }
        other if other.starts_with('-') => Err(anyhow::anyhow!("Unknown option: {}", other)),
        _ => Err(anyhow::anyhow!(
            "Unknown command: {}. Valid commands: run, pull, push, sync, prune",
            args[0]
        )),
    }
}

fn parse_prune_local_options(args: &mut Vec<String>) -> Result<prune::PruneLocalOptions> {
    let mut target = None;
    let mut artifact = None;
    let mut no_checkpoints = false;
    let mut no_artifacts = false;
    let mut dry_run = false;
    let mut assume_yes = false;

    while !args.is_empty() {
        match args[0].as_str() {
            "--artifact" => {
                args.remove(0);
                artifact = Some(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--artifact requires a value"))?
                        .clone(),
                );
                args.remove(0);
            }
            "--no-checkpoints" => {
                no_checkpoints = true;
                args.remove(0);
            }
            "--no-artifacts" => {
                no_artifacts = true;
                args.remove(0);
            }
            "-n" | "--dry-run" => {
                dry_run = true;
                args.remove(0);
            }
            "-y" | "--yes" => {
                assume_yes = true;
                args.remove(0);
            }
            s if s.starts_with('-') => {
                return Err(anyhow::anyhow!("Unknown option: {}", s));
            }
            _ => {
                if target.is_none() {
                    target = Some(args.remove(0));
                } else {
                    return Err(anyhow::anyhow!("Unexpected argument: {}", args[0]));
                }
            }
        }
    }

    Ok(prune::PruneLocalOptions {
        target,
        artifact,
        no_checkpoints,
        no_artifacts,
        dry_run,
        assume_yes,
    })
}

fn parse_run_options(args: &mut Vec<String>) -> Result<run::RunOptions> {
    let mut provider = None;
    let mut instance_id = None;
    let mut python_version = "3.12".to_string();
    let mut skip_tmux = false;
    let mut exclude = Vec::new();
    let mut command = Vec::new();

    while !args.is_empty() {
        match args[0].as_str() {
            "--" => {
                args.remove(0);
                command.append(args);
                break;
            }
            "--provider" => {
                args.remove(0);
                provider = Some(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--provider requires a value"))?
                        .clone(),
                );
                args.remove(0);
            }
            "--instance" => {
                args.remove(0);
                instance_id = Some(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--instance requires a value"))?
                        .clone(),
                );
                args.remove(0);
            }
            "--python" => {
                args.remove(0);
                python_version = args
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("--python requires a value"))?
                    .clone();
                args.remove(0);
            }
            "--skip-tmux" => {
                skip_tmux = true;
                args.remove(0);
            }
            "--exclude" => {
                args.remove(0);
                exclude.push(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--exclude requires a value"))?
                        .clone(),
                );
                args.remove(0);
            }
            _ => {
                command.append(args);
                break;
            }
        }
    }

    Ok(run::RunOptions {
        provider,
        instance_id,
        python_version,
        skip_tmux,
        exclude,
        command,
    })
}

fn parse_sync_options(args: &mut Vec<String>) -> Result<SyncOptions> {
    let mut target = None;
    let mut force = false;
    let mut dry_run = false;

    while !args.is_empty() {
        match args[0].as_str() {
            "-f" | "--force" => {
                force = true;
                args.remove(0);
            }
            "-n" | "--dry-run" => {
                dry_run = true;
                args.remove(0);
            }
            s if s.starts_with('-') => {
                return Err(anyhow::anyhow!("Unknown option: {}", s));
            }
            _ => {
                if target.is_none() {
                    target = Some(args.remove(0));
                } else {
                    return Err(anyhow::anyhow!("Unexpected argument: {}", args[0]));
                }
            }
        }
    }

    Ok(SyncOptions {
        target,
        force,
        dry_run,
    })
}

fn render(app: &mut App, frame: &mut Frame) {
    match app.view {
        View::List => match app.view_mode {
            ViewMode::Runs => render_runs_list(app, frame),
            ViewMode::Artifacts => render_artifacts_list(app, frame),
            ViewMode::Infra => render_infra_dashboard(app, frame),
        },
        View::RunDetail => render_run_detail(app, frame),
        View::Focused => render_focused(app, frame),
        View::ConfigFull => render_config_full(app, frame),
        View::Compare => render_compare_view(app, frame),
        View::InfraList => render_infra_dashboard(app, frame),
        View::InfraConfig => render_infra_config(app, frame),
        View::S3Config => render_s3_config(app, frame),
        View::ArtifactDetail => render_artifact_detail(app, frame),
    }

    if app.show_delete_confirm {
        render_delete_confirm(app, frame);
    }
    if app.show_terminate_confirm {
        render_terminate_confirm(app, frame);
    }
    if app.show_complete_confirm {
        render_complete_confirm(app, frame);
    }
    if app.pull_run_modal_open {
        render_pull_run_modal(app, frame);
    }
    if app.note_modal_open {
        render_note_modal(app, frame);
    }
    if app.move_run_modal_open {
        render_move_run_modal(app, frame);
    }
    if app.filter_modal_open {
        render_filter_modal(app, frame);
    }
    if app.checkpoint_download_modal {
        render_checkpoint_download_modal(app, frame);
    }
    if app.help_overlay_open {
        render_help_overlay(app, frame);
    }
}

fn render_runs_list(app: &mut App, frame: &mut Frame) {
    let area = frame.area();
    let entries = app.list_entries();

    let items: Vec<ListItem> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let is_selected = i == app.selected_list_item;

            match entry {
                ListEntry::Project { name } => {
                    let is_expanded = app.expanded_projects.contains(name);
                    let icon = if is_expanded { "▼ " } else { "▶ " };
                    let is_archived = app.archived_projects.contains(name);

                    let matching = |r: &&Run| {
                        r.project.as_deref().unwrap_or("(no project)") == name
                            && (!app.hide_completed || r.is_running())
                            && (app.config_filters.is_empty()
                                || run_matches_filters(r, &app.config_filters))
                    };
                    let run_count = app.runs.iter().filter(matching).count();
                    let has_running = app.runs.iter().filter(matching).any(|r| r.is_running());

                    let name_style = if is_selected {
                        Style::default().fg(NEON_MAGENTA).bold()
                    } else if is_archived {
                        Style::default().fg(Color::DarkGray).bold()
                    } else if has_running {
                        Style::default().fg(NEON_GREEN).bold()
                    } else {
                        Style::default().fg(NEON_CYAN).bold()
                    };

                    let mut spans = vec![
                        Span::styled(icon, Style::default().fg(NEON_MAGENTA)),
                        Span::styled(name.clone(), name_style),
                        Span::styled(
                            format!("  ({} runs)", run_count),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ];
                    if is_archived {
                        spans.push(Span::styled(
                            "  [archived]",
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    ListItem::new(Line::from(spans))
                }
                ListEntry::Run { run_index } => {
                    let run = &app.runs[*run_index];
                    let is_running = run.is_running();

                    let (status_icon, status_color) = if is_running {
                        ("● ", NEON_GREEN)
                    } else {
                        ("○ ", Color::DarkGray)
                    };

                    let name_style = if is_selected {
                        Style::default().fg(NEON_CYAN).bold()
                    } else if is_running {
                        Style::default().fg(NEON_GREEN)
                    } else {
                        Style::default().fg(Color::Gray)
                    };

                    let start_str = run
                        .start_time
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "—".to_string());

                    let end_str = if is_running {
                        "running...".to_string()
                    } else {
                        run.end_time
                            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_else(|| "—".to_string())
                    };

                    let time_style = if is_running {
                        Style::default().fg(NEON_YELLOW)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };

                    let is_compared = app.compared_runs.contains(run_index);
                    let check = if is_compared { "✓" } else { " " };
                    let check_color = if is_compared {
                        let ci = app
                            .compared_runs
                            .iter()
                            .position(|i| i == run_index)
                            .unwrap_or(0);
                        COMPARE_COLORS[ci % COMPARE_COLORS.len()]
                    } else {
                        Color::DarkGray
                    };

                    let run_display_name = run.display_name();
                    let is_starred = app.starred_runs.contains(&run_display_name);
                    let star = if is_starred { "★" } else { " " };
                    let star_color = if is_starred {
                        NEON_YELLOW
                    } else {
                        Color::DarkGray
                    };

                    let has_note = app.run_notes.contains_key(&run_display_name);
                    let note_icon = if has_note { "✎" } else { " " };
                    let note_color = if has_note { NEON_CYAN } else { Color::DarkGray };

                    let spans = vec![
                        Span::styled(check, Style::default().fg(check_color)),
                        Span::styled(" ", Style::default()),
                        Span::styled(star, Style::default().fg(star_color)),
                        Span::styled(note_icon, Style::default().fg(note_color)),
                        Span::styled(" └─ ", Style::default().fg(DIM_CYAN)),
                        Span::styled(status_icon, Style::default().fg(status_color)),
                        Span::styled(run.name.clone(), name_style),
                        Span::styled("  ", Style::default()),
                        Span::styled(start_str, Style::default().fg(Color::DarkGray)),
                        Span::styled(" → ", Style::default().fg(DIM_CYAN)),
                        Span::styled(end_str, time_style),
                    ];
                    ListItem::new(Line::from(spans))
                }
            }
        })
        .collect();

    let mut title_spans = vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("[Runs]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Artifacts", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Infra", Style::default().fg(Color::DarkGray)),
        Span::styled(" ", Style::default()),
    ];
    if let Some(status) = &app.s3_pull_status {
        let color = if status.starts_with("Pull failed")
            || status.starts_with("S3 not")
            || status.starts_with("List failed")
        {
            NEON_MAGENTA
        } else if status.starts_with("Pulled") || status.starts_with("No remote") {
            NEON_GREEN
        } else {
            NEON_YELLOW
        };
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        title_spans.push(Span::styled(status.clone(), Style::default().fg(color)));
    }

    if app.hide_completed {
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        title_spans.push(Span::styled(
            "running only",
            Style::default().fg(NEON_GREEN),
        ));
    }

    if app.show_archived {
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        title_spans.push(Span::styled(
            "showing archived",
            Style::default().fg(NEON_YELLOW),
        ));
    }

    if !app.config_filters.is_empty() {
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        let filter_str = app
            .config_filters
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" & ");
        title_spans.push(Span::styled(filter_str, Style::default().fg(NEON_YELLOW)));
    }

    if !app.search_query.is_empty() || app.search_editing {
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        title_spans.push(Span::styled("/", Style::default().fg(NEON_CYAN)));
        title_spans.push(Span::styled(
            app.search_query.clone(),
            Style::default().fg(NEON_YELLOW),
        ));
        if app.search_editing {
            title_spans.push(Span::styled("█", Style::default().fg(NEON_CYAN)));
        }
    }

    let list = List::new(items)
        .block(
            Block::default()
                .title(Line::from(title_spans))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut app.list_state);

    let mut help_spans = vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] open  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] expand  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Space", Style::default().fg(NEON_CYAN)),
        Span::styled("] compare  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("/", Style::default().fg(NEON_YELLOW)),
        Span::styled("] search  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("f", Style::default().fg(NEON_YELLOW)),
        Span::styled("] filter  ", Style::default().fg(Color::DarkGray)),
    ];
    if app.compared_runs.len() >= 2 {
        help_spans.extend(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("v", Style::default().fg(NEON_YELLOW)),
            Span::styled(
                format!("] view {}  ", app.compared_runs.len()),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
    }
    help_spans.extend(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("?", Style::default().fg(NEON_YELLOW)),
        Span::styled("] help", Style::default().fg(Color::DarkGray)),
    ]);
    let help = Line::from(help_spans);
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn format_time_ago(secs: u64) -> String {
    match secs {
        0..=59 => format!("{}s ago", secs),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86399 => {
            let h = secs / 3600;
            let m = (secs % 3600) / 60;
            if m == 0 {
                format!("{}h ago", h)
            } else {
                format!("{}h {}m ago", h, m)
            }
        }
        _ => {
            let d = secs / 86400;
            format!("{}d ago", d)
        }
    }
}

fn render_run_detail(app: &mut App, frame: &mut Frame) {
    let area = frame.area();

    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    // Split main content horizontally if config panel is shown
    let config_width = 35u16;
    let (grid_area, config_area) = if app.show_config && run.config.is_some() {
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(config_width)])
            .split(chunks[1]);
        (h_chunks[0], Some(h_chunks[1]))
    } else {
        (chunks[1], None)
    };

    // Calculate scroll info for header
    let card_width = 40u16;
    let card_height = 12u16;
    let cols = (grid_area.width / card_width).max(1) as usize;
    let cards = app.cards();
    let total_cards = cards.len();
    let total_rows = total_cards.div_ceil(cols);
    let visible_rows = (grid_area.height / card_height) as usize;
    let max_scroll = total_rows.saturating_sub(visible_rows);
    let scroll = app.scroll_offset.min(max_scroll);

    // Header with scroll indicator
    let scroll_indicator = if total_rows > visible_rows {
        let has_above = scroll > 0;
        let has_below = scroll < max_scroll;
        match (has_above, has_below) {
            (true, true) => " [↑↓ more]".to_string(),
            (true, false) => " [↑ more above]".to_string(),
            (false, true) => " [↓ more below]".to_string(),
            (false, false) => String::new(),
        }
    } else {
        String::new()
    };

    let total_examples: usize = run.examples.values().map(|g| g.len()).sum();
    let latest_step = run
        .metrics
        .values()
        .flat_map(|pts| pts.last())
        .map(|p| p.step)
        .max()
        .unwrap_or(0);
    let mut header_spans = vec![
        Span::styled("◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled(run.display_name(), Style::default().fg(NEON_CYAN).bold()),
        Span::styled("  │  ", Style::default().fg(DIM_CYAN)),
        Span::styled(
            format!("{}", latest_step),
            Style::default().fg(NEON_MAGENTA),
        ),
        Span::styled(" steps  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", run.metrics.len()),
            Style::default().fg(NEON_GREEN),
        ),
        Span::styled(" charts  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", total_examples),
            Style::default().fg(NEON_YELLOW),
        ),
        Span::styled(" examples", Style::default().fg(Color::DarkGray)),
    ];
    if !run.checkpoints.is_empty() {
        header_spans.push(Span::styled("  ", Style::default().fg(Color::DarkGray)));
        header_spans.push(Span::styled(
            format!("{}", run.checkpoints.len()),
            Style::default().fg(NEON_MAGENTA),
        ));
        header_spans.push(Span::styled(
            " checkpoints",
            Style::default().fg(Color::DarkGray),
        ));
    }
    let artifact_count = app.run_artifacts(run).len();
    if artifact_count > 0 {
        header_spans.push(Span::styled("  ", Style::default().fg(Color::DarkGray)));
        header_spans.push(Span::styled(
            format!("{}", artifact_count),
            Style::default().fg(NEON_YELLOW),
        ));
        header_spans.push(Span::styled(
            " artifacts",
            Style::default().fg(Color::DarkGray),
        ));
    }
    header_spans.push(Span::styled(
        &scroll_indicator,
        Style::default().fg(NEON_MAGENTA),
    ));
    if let Some(status) = &app.s3_pull_status {
        let color = if status.starts_with("Pull failed") || status.starts_with("S3 not") {
            NEON_MAGENTA
        } else if status.starts_with("Pulled") {
            NEON_GREEN
        } else {
            NEON_YELLOW
        };
        header_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        header_spans.push(Span::styled(status.clone(), Style::default().fg(color)));
    }
    if let Some(elapsed) = run
        .last_modified()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
    {
        header_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        header_spans.push(Span::styled(
            format!("updated {}", format_time_ago(elapsed.as_secs())),
            Style::default().fg(Color::DarkGray),
        ));
    }
    if let Some(note) = app.run_notes.get(&run.display_name()) {
        let display_note = if note.len() > 60 {
            format!("{}...", &note[..57])
        } else {
            note.clone()
        };
        header_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        header_spans.push(Span::styled("✎ ", Style::default().fg(NEON_CYAN)));
        header_spans.push(Span::styled(
            display_note,
            Style::default().fg(Color::White),
        ));
    }
    let header_text = Line::from(header_spans);
    let header = Paragraph::new(header_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN)),
    );
    frame.render_widget(header, chunks[0]);

    // Cards grid
    render_cards_grid(app, frame, grid_area, &cards);

    // Config panel (if shown)
    let run_config_cloned = run.config.clone();
    let has_sys_metrics = run.metrics.keys().any(|k| k.starts_with("sys/"));
    let is_running = run.is_running();
    if let Some(config_area) = config_area
        && let Some(ref config) = run_config_cloned
    {
        render_config_panel(frame, config_area, config, &mut app.config_panel_scroll);
    }

    // Footer with styled keys
    let config_hint = if app.show_config { "hide" } else { "config" };
    let mut footer_spans = vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("←→", Style::default().fg(NEON_CYAN)),
        Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_GREEN)),
        Span::styled("] focus  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("c", Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("] {}  ", config_hint),
            Style::default().fg(Color::DarkGray),
        ),
    ];
    if has_sys_metrics {
        let sys_hint = if app.show_system_metrics {
            "train"
        } else {
            "system"
        };
        footer_spans.extend([
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("S", Style::default().fg(NEON_CYAN)),
            Span::styled(
                format!("] {}  ", sys_hint),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
    }
    if let Some(Card::Chart { name }) = cards.get(app.selected_card) {
        let t_hint = match app.chart_x_axis(name) {
            XAxis::Step => "time x",
            XAxis::Timestamp => "step x",
        };
        footer_spans.extend([
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("t", Style::default().fg(NEON_CYAN)),
            Span::styled(
                format!("] {}  ", t_hint),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
    }
    if app.show_config {
        footer_spans.extend([
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("JK", Style::default().fg(NEON_CYAN)),
            Span::styled("] scroll config  ", Style::default().fg(Color::DarkGray)),
        ]);
    }
    if is_running {
        footer_spans.extend([
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("m", Style::default().fg(NEON_YELLOW)),
            Span::styled("] mark done  ", Style::default().fg(Color::DarkGray)),
        ]);
    }
    footer_spans.extend([
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("[]", Style::default().fg(NEON_YELLOW)),
        Span::styled("] run ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}/{}  ", app.selected_run + 1, app.runs.len()),
            Style::default().fg(NEON_YELLOW),
        ),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("?", Style::default().fg(NEON_YELLOW)),
        Span::styled("] help", Style::default().fg(Color::DarkGray)),
    ]);
    let footer = Line::from(footer_spans);
    frame.render_widget(Paragraph::new(footer), chunks[2]);
}

fn render_cards_grid(app: &App, frame: &mut Frame, area: Rect, cards: &[Card]) {
    let Some(run) = app.current_run() else { return };

    if cards.is_empty() {
        let msg = if run.data_loaded {
            "No data"
        } else {
            "Loading..."
        };
        frame.render_widget(Paragraph::new(msg), area);
        return;
    }

    // Calculate grid layout
    let card_width = 40u16;
    let card_height = 12u16;
    let cols = (area.width / card_width).max(1) as usize;

    let total_rows = cards.len().div_ceil(cols);
    let visible_rows = (area.height / card_height) as usize;
    let max_scroll = total_rows.saturating_sub(visible_rows);
    let scroll = app.scroll_offset.min(max_scroll);

    for (i, card) in cards.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;

        // Skip rows above scroll offset
        if row < scroll {
            continue;
        }

        let visible_row = row - scroll;
        let x = area.x + (col as u16) * card_width;
        let y = area.y + (visible_row as u16) * card_height;

        // Skip if outside visible area
        if y + card_height > area.bottom() {
            continue;
        }

        let card_area = Rect::new(x, y, card_width.min(area.right() - x), card_height);
        let is_selected = i == app.selected_card;

        match card {
            Card::Chart { name } => {
                if let Some(points) = run.metrics.get(name) {
                    let x_axis = app.chart_x_axis(name);
                    render_chart(frame, card_area, name, points, is_selected, x_axis);
                }
            }
            Card::Examples { name } => {
                if let Some(examples) = run.examples.get(name) {
                    render_examples_card(frame, card_area, name, examples, is_selected);
                }
            }
            Card::ConfusionMatrix { name } => {
                if let Some(series) = run.confusion_matrices.get(name) {
                    let idx = app
                        .confusion_step_idx
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| series.len().saturating_sub(1));
                    if let Some(point) = series.get(idx) {
                        render_confusion_matrix_card(frame, card_area, name, point, is_selected);
                    }
                }
            }
            Card::Checkpoints => {
                render_checkpoints_card(
                    frame,
                    card_area,
                    &run.checkpoints,
                    &run.metrics,
                    is_selected,
                );
            }
            Card::Artifacts => {
                let run_artifacts = app.run_artifacts(run);
                render_artifacts_card(frame, card_area, &run_artifacts, is_selected);
            }
        }
    }
}

fn render_artifacts_card(frame: &mut Frame, area: Rect, artifacts: &[&Artifact], selected: bool) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let block = Block::default()
        .title(Span::styled(
            format!(" Artifacts ({})", artifacts.len()),
            title_style,
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let max_lines = inner.height as usize;
    let lines: Vec<Line> = artifacts
        .iter()
        .take(max_lines)
        .map(|a| {
            let icon = if a.content_type == "directory" {
                "📁"
            } else {
                "📄"
            };
            Line::from(vec![
                Span::styled(format!("{} ", icon), Style::default()),
                Span::styled(&a.name, Style::default().fg(Color::White)),
                Span::styled(
                    format!("  ({})", a.display_size()),
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_comparison_chart(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    run_data: &[CompareRunData],
    selected: bool,
    x_axis: XAxis,
) {
    use ratatui::symbols::Marker;
    use ratatui::widgets::{Axis, Chart, Dataset, GraphType, LegendPosition};

    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let non_empty: Vec<&CompareRunData> = run_data
        .iter()
        .filter(|(_, _, d, _)| !d.is_empty())
        .collect();

    let display_title = match x_axis {
        XAxis::Step => title.to_string(),
        XAxis::Timestamp => format!("{} (time)", title),
    };

    if non_empty.is_empty() {
        let block = Block::default()
            .title(Span::styled(display_title.clone(), title_style))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color));
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No data",
                Style::default().fg(Color::DarkGray),
            ))
            .block(block),
            area,
        );
        return;
    }

    let all_scalar = non_empty.iter().all(|(_, _, _, is_scalar)| *is_scalar);
    if all_scalar {
        render_scalar_bar_compare(frame, area, &display_title, run_data, selected);
        return;
    }

    let x_min = non_empty
        .iter()
        .filter_map(|(_, _, d, _)| d.first().map(|p| p.0))
        .fold(f64::INFINITY, f64::min);
    let x_max = non_empty
        .iter()
        .filter_map(|(_, _, d, _)| d.last().map(|p| p.0))
        .fold(f64::NEG_INFINITY, f64::max);
    let y_data_min = non_empty
        .iter()
        .flat_map(|(_, _, d, _)| d.iter().map(|p| p.1))
        .fold(f64::INFINITY, f64::min);
    let y_data_max = non_empty
        .iter()
        .flat_map(|(_, _, d, _)| d.iter().map(|p| p.1))
        .fold(f64::NEG_INFINITY, f64::max);

    let y_range = (y_data_max - y_data_min).max(0.001);
    let y_min = y_data_min - y_range * 0.1;
    let y_max = y_data_max + y_range * 0.1;

    let max_points = (area.width as usize) * 2;
    let downsampled: Vec<CompareRunData> = run_data
        .iter()
        .filter(|(_, _, d, _)| !d.is_empty())
        .map(|(name, color, data, is_scalar)| {
            (
                name.clone(),
                *color,
                lttb_downsample(data, max_points),
                *is_scalar,
            )
        })
        .collect();

    let datasets: Vec<Dataset> = downsampled
        .iter()
        .map(|(name, color, data, is_scalar)| {
            let (marker, graph_type) = if *is_scalar {
                (Marker::Block, GraphType::Scatter)
            } else {
                (Marker::Braille, GraphType::Line)
            };
            Dataset::default()
                .name(name.as_str())
                .marker(marker)
                .graph_type(graph_type)
                .style(Style::default().fg(*color))
                .data(data)
        })
        .collect();

    let axis_style = Style::default().fg(DIM_CYAN);
    let label_style = Style::default().fg(Color::DarkGray);

    let x_span = x_max - x_min;
    let num_x_ticks = 5;
    let x_labels: Vec<Span> = (0..num_x_ticks)
        .map(|i| {
            let v = x_min + x_span * i as f64 / (num_x_ticks - 1) as f64;
            Span::styled(format_x_label(v, x_span, x_axis), label_style)
        })
        .collect();

    let num_y_ticks = 5;
    let y_range = y_data_max - y_data_min;
    let y_labels: Vec<Span> = (0..num_y_ticks)
        .map(|i| {
            let v = y_data_min + y_range * i as f64 / (num_y_ticks - 1) as f64;
            Span::styled(format_y_label(v, y_range), label_style)
        })
        .collect();

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(display_title, title_style))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color)),
        )
        .x_axis(
            Axis::default()
                .style(axis_style)
                .bounds([x_min, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .style(axis_style)
                .bounds([y_min, y_max])
                .labels(y_labels),
        )
        .legend_position(Some(LegendPosition::TopRight));

    frame.render_widget(chart, area);
}

fn render_scalar_bar_compare(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    run_data: &[CompareRunData],
    selected: bool,
) {
    use ratatui::widgets::{Bar, BarChart, BarGroup};

    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let block = Block::default()
        .title(Span::styled(title.to_string(), title_style))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let entries: Vec<(String, Color, f64)> = run_data
        .iter()
        .filter_map(|(name, color, data, _)| data.last().map(|p| (name.clone(), *color, p.1)))
        .collect();

    if entries.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No data",
                Style::default().fg(Color::DarkGray),
            ))
            .block(block),
            area,
        );
        return;
    }

    let raw_min = entries
        .iter()
        .map(|(_, _, v)| *v)
        .fold(f64::INFINITY, f64::min);
    let raw_max = entries
        .iter()
        .map(|(_, _, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (raw_max - raw_min).abs().max(1e-9);
    let pad = range * 0.15;
    let scale_max = 10_000.0;

    let bars: Vec<Bar> = entries
        .iter()
        .map(|(name, color, value)| {
            let normalized = (value - raw_min + pad) / (range + pad);
            let scaled = (normalized * scale_max).max(1.0) as u64;
            let label_text = if name.chars().count() > 10 {
                let head: String = name.chars().take(9).collect();
                format!("{}…", head)
            } else {
                name.clone()
            };
            Bar::default()
                .label(Line::from(label_text))
                .value(scaled)
                .text_value(format_scalar_value(*value))
                .style(Style::default().fg(*color))
                .value_style(Style::default().fg(*color).bold())
        })
        .collect();

    let inner_width = block.inner(area).width;
    let n = bars.len() as u16;
    let gap: u16 = 1;
    let total_gap = gap * n.saturating_sub(1);
    let bar_width = if n == 0 {
        1
    } else {
        ((inner_width.saturating_sub(total_gap)) / n).max(1)
    };

    let chart = BarChart::default()
        .block(block)
        .data(BarGroup::default().bars(&bars))
        .bar_width(bar_width)
        .bar_gap(gap)
        .label_style(Style::default().fg(Color::DarkGray));

    frame.render_widget(chart, area);
}

fn render_compare_view(app: &mut App, frame: &mut Frame) {
    let area = frame.area();
    let cards = app.compare_cards();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    let mut header_spans: Vec<Span> = vec![
        Span::styled("◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Compare: ", Style::default().fg(NEON_CYAN).bold()),
    ];
    for (i, &run_idx) in app.compared_runs.iter().enumerate() {
        if let Some(run) = app.runs.get(run_idx) {
            let color = COMPARE_COLORS[i % COMPARE_COLORS.len()];
            if i > 0 {
                header_spans.push(Span::styled("  ", Style::default()));
            }
            header_spans.push(Span::styled("●", Style::default().fg(color)));
            header_spans.push(Span::styled(
                format!(" {}", run.display_name()),
                Style::default().fg(color),
            ));
        }
    }
    if let Some(status) = &app.s3_pull_status {
        let color = if status.contains("failed")
            || status.contains("error")
            || status.contains("not configured")
        {
            NEON_MAGENTA
        } else if status.contains("Pulled") {
            NEON_GREEN
        } else {
            NEON_YELLOW
        };
        header_spans.push(Span::styled("  ", Style::default()));
        header_spans.push(Span::styled(status.as_str(), Style::default().fg(color)));
    }
    frame.render_widget(Paragraph::new(Line::from(header_spans)), chunks[0]);

    let has_any_config = app
        .compared_runs
        .iter()
        .any(|&ri| app.runs.get(ri).and_then(|r| r.config.as_ref()).is_some());
    let config_width = 40u16;
    let (grid_area, config_area) = if app.show_config && has_any_config {
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(40),
                Constraint::Length(config_width * app.compared_runs.len() as u16),
            ])
            .split(chunks[1]);
        (h_chunks[0], Some(h_chunks[1]))
    } else {
        (chunks[1], None)
    };

    let card_width = 40u16;
    let card_height = 12u16;
    let cols = (grid_area.width / card_width).max(1) as usize;
    let total_rows = cards.len().div_ceil(cols);
    let visible_rows = (grid_area.height / card_height) as usize;
    let max_scroll = total_rows.saturating_sub(visible_rows);
    let scroll = app.scroll_offset.min(max_scroll);

    for (i, card) in cards.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;

        if row < scroll {
            continue;
        }

        let visible_row = row - scroll;
        let x = grid_area.x + (col as u16) * card_width;
        let y = grid_area.y + (visible_row as u16) * card_height;

        if y + card_height > grid_area.bottom() {
            continue;
        }

        let card_area = Rect::new(x, y, card_width.min(grid_area.right() - x), card_height);
        let is_selected = i == app.selected_card;

        match card {
            Card::Chart { name } => {
                let x_axis = app.chart_x_axis(name);
                let run_data: Vec<CompareRunData> = app
                    .compared_runs
                    .iter()
                    .enumerate()
                    .filter_map(|(ci, &run_idx)| {
                        let run = app.runs.get(run_idx)?;
                        let color = COMPARE_COLORS[ci % COMPARE_COLORS.len()];
                        let pts = run.metrics.get(name);
                        let is_scalar = pts.map(|p| is_scalar_series(p)).unwrap_or(true);
                        let data: Vec<(f64, f64)> =
                            pts.map(|p| points_to_xy(p, x_axis)).unwrap_or_default();
                        Some((run.display_name(), color, data, is_scalar))
                    })
                    .collect();
                render_comparison_chart(frame, card_area, name, &run_data, is_selected, x_axis);
            }
            Card::Examples { name } => {
                render_compare_examples_card(frame, card_area, name, app, is_selected);
            }
            _ => {}
        }
    }

    if let Some(config_area) = config_area {
        let configs: Vec<Option<&serde_json::Value>> = app
            .compared_runs
            .iter()
            .map(|&ri| app.runs.get(ri).and_then(|r| r.config.as_ref()))
            .collect();
        let diff_keys = collect_differing_keys(&configs);

        let col_constraints: Vec<Constraint> = app
            .compared_runs
            .iter()
            .map(|_| Constraint::Ratio(1, app.compared_runs.len() as u32))
            .collect();
        let col_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_constraints)
            .split(config_area);
        for (i, &run_idx) in app.compared_runs.iter().enumerate() {
            let color = COMPARE_COLORS[i % COMPARE_COLORS.len()];
            if let Some(run) = app.runs.get(run_idx) {
                if let Some(config) = &run.config {
                    let mut lines: Vec<Line> = Vec::new();
                    render_json_value(config, 0, &mut lines, Some(&diff_keys), "");
                    let total_lines = lines.len();
                    let visible = col_chunks[i].height.saturating_sub(2) as usize;
                    let max_scroll = total_lines.saturating_sub(visible);
                    app.config_panel_scroll = app.config_panel_scroll.min(max_scroll);
                    let scroll = app.config_panel_scroll as u16;
                    let block = Block::default()
                        .title(Span::styled(
                            format!(" {} ", run.display_name()),
                            Style::default().fg(color).bold(),
                        ))
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(DIM_CYAN));
                    let paragraph = Paragraph::new(lines).block(block).scroll((scroll, 0));
                    frame.render_widget(paragraph, col_chunks[i]);
                } else {
                    let block = Block::default()
                        .title(Span::styled(
                            format!(" {} ", run.display_name()),
                            Style::default().fg(color).bold(),
                        ))
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(DIM_CYAN));
                    let paragraph = Paragraph::new(Span::styled(
                        "No config",
                        Style::default().fg(Color::DarkGray),
                    ))
                    .block(block);
                    frame.render_widget(paragraph, col_chunks[i]);
                }
            }
        }
    }

    let config_hint = if app.show_config { "hide" } else { "config" };
    let sys_hint = if app.show_system_metrics {
        "train"
    } else {
        "system"
    };
    let mut footer_spans: Vec<Span> = vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("←→", Style::default().fg(NEON_CYAN)),
        Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] focus  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("S", Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("] {}  ", sys_hint),
            Style::default().fg(Color::DarkGray),
        ),
    ];
    if let Some(Card::Chart { name }) = cards.get(app.selected_card) {
        let t_hint = match app.chart_x_axis(name) {
            XAxis::Step => "time x",
            XAxis::Timestamp => "step x",
        };
        footer_spans.extend([
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("t", Style::default().fg(NEON_CYAN)),
            Span::styled(
                format!("] {}  ", t_hint),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
    }
    footer_spans.extend([
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("c", Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("] {}  ", config_hint),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("p", Style::default().fg(NEON_CYAN)),
        Span::styled("] pull", Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(Line::from(footer_spans)), chunks[2]);
}

fn render_compare_examples_card(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    app: &App,
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let max_lines = area.height.saturating_sub(4) as usize;
    let mut lines: Vec<Line> = Vec::new();

    for (i, &run_idx) in app.compared_runs.iter().enumerate() {
        if lines.len() >= max_lines {
            break;
        }
        let color = COMPARE_COLORS[i % COMPARE_COLORS.len()];
        if let Some(run) = app.runs.get(run_idx) {
            let count = run.examples.get(name).map(|g| g.len()).unwrap_or(0);
            let label = if count > 0 {
                format!("{}: {} examples", run.display_name(), count)
            } else {
                format!("{}: —", run.display_name())
            };
            lines.push(Line::from(vec![
                Span::styled("● ", Style::default().fg(color)),
                Span::styled(label, Style::default().fg(Color::Gray)),
            ]));
        }
    }

    if lines.len() < max_lines
        && let Some(example) = app
            .compared_runs
            .iter()
            .filter_map(|&idx| app.runs.get(idx))
            .filter_map(|run| run.examples.get(name))
            .filter_map(|g| g.last())
            .next()
    {
        lines.push(Line::from(""));
        let preview: String = example
            .prompts
            .first()
            .map(|s| s.chars().take(40).collect::<String>())
            .unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled("Q: ", Style::default().fg(NEON_YELLOW).bold()),
            Span::styled(format!("{}...", preview), Style::default().fg(Color::White)),
        ]));
    }

    let title = Line::from(vec![
        Span::styled(format!("{} ", name), title_style),
        Span::styled("(examples)", Style::default().fg(Color::DarkGray)),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(ratatui::widgets::Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn render_config_panel(
    frame: &mut Frame,
    area: Rect,
    config: &serde_json::Value,
    scroll: &mut usize,
) {
    let mut lines: Vec<Line> = Vec::new();
    render_json_value(config, 0, &mut lines, None, "");

    let total_lines = lines.len();
    let visible = area.height.saturating_sub(2) as usize;
    let max_scroll = total_lines.saturating_sub(visible);
    *scroll = (*scroll).min(max_scroll);
    let scroll = *scroll as u16;

    let has_above = scroll > 0;
    let has_below = (scroll as usize) < max_scroll;
    let scroll_hint = match (has_above, has_below) {
        (true, true) => " ↑↓ ",
        (true, false) => " ↑ ",
        (false, true) => " ↓ ",
        _ => "",
    };

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" CONFIG ", Style::default().fg(NEON_CYAN).bold()),
            Span::styled(scroll_hint, Style::default().fg(Color::DarkGray)),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let paragraph = Paragraph::new(lines).block(block).scroll((scroll, 0));

    frame.render_widget(paragraph, area);
}

fn render_config_full(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    let Some(config) = &run.config else {
        frame.render_widget(Paragraph::new("No config available"), area);
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)])
        .split(area);

    let mut lines: Vec<Line> = Vec::new();
    render_json_value_full(config, 0, &mut lines);

    let total_lines = lines.len();
    let visible = chunks[0].height.saturating_sub(2) as usize;
    let cursor = app.config_cursor.min(total_lines.saturating_sub(1));

    // Highlight the cursor line
    if let Some(line) = lines.get_mut(cursor) {
        let mut spans = vec![Span::styled("▶ ", Style::default().fg(NEON_CYAN))];
        spans.extend(line.spans.clone());
        *line = Line::from(spans).patch_style(Style::default().bg(Color::Rgb(40, 55, 70)));
    }

    // Auto-scroll to keep cursor visible
    let scroll = if cursor < visible / 2 {
        0
    } else {
        (cursor - visible / 2).min(total_lines.saturating_sub(visible))
    } as u16;

    let mut title_spans = vec![
        Span::styled(" CONFIG ", Style::default().fg(NEON_CYAN).bold()),
        Span::styled("— ", Style::default().fg(DIM_CYAN)),
        Span::styled(run.display_name(), Style::default().fg(NEON_MAGENTA)),
        Span::styled(" ", Style::default()),
    ];

    if let Some(t) = app.config_copied_at
        && t.elapsed() < Duration::from_secs(2)
    {
        title_spans.push(Span::styled(
            " Copied! ",
            Style::default().fg(NEON_GREEN).bold(),
        ));
    }

    let block = Block::default()
        .title(Line::from(title_spans))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let paragraph = Paragraph::new(lines).block(block).scroll((scroll, 0));

    frame.render_widget(paragraph, chunks[0]);

    let footer = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("y", Style::default().fg(NEON_GREEN)),
        Span::styled("] copy value  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("?", Style::default().fg(NEON_YELLOW)),
        Span::styled("] help", Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[1]);
}

fn render_json_value_full(value: &serde_json::Value, indent: usize, lines: &mut Vec<Line>) {
    let pad = "  ".repeat(indent);
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}:", key), Style::default().fg(NEON_MAGENTA)),
                        ]));
                        render_json_value_full(val, indent + 1, lines);
                    }
                    _ => {
                        let val_str = format_json_primitive(val);
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                            Span::styled(val_str, Style::default().fg(Color::White)),
                        ]));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("[{}]:", i), Style::default().fg(NEON_YELLOW)),
                        ]));
                        render_json_value_full(val, indent + 1, lines);
                    }
                    _ => {
                        let val_str = format_json_primitive(val);
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("[{}]: ", i), Style::default().fg(NEON_YELLOW)),
                            Span::styled(val_str, Style::default().fg(Color::White)),
                        ]));
                    }
                }
            }
        }
        _ => {
            let val_str = format_json_primitive(value);
            lines.push(Line::from(Span::styled(
                format!("{}{}", pad, val_str),
                Style::default().fg(Color::White),
            )));
        }
    }
}

fn flatten_config_values(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (_, val) in map {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        out.push(serde_json::to_string_pretty(val).unwrap_or_default());
                        flatten_config_values(val, out);
                    }
                    _ => {
                        out.push(format_json_primitive(val));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for val in arr {
                match val {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        out.push(serde_json::to_string_pretty(val).unwrap_or_default());
                        flatten_config_values(val, out);
                    }
                    _ => {
                        out.push(format_json_primitive(val));
                    }
                }
            }
        }
        _ => {
            out.push(format_json_primitive(value));
        }
    }
}

fn copy_to_clipboard(text: &str) {
    use std::process::{Command, Stdio};
    if let Ok(mut child) = Command::new("pbcopy").stdin(Stdio::piped()).spawn() {
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
        }
        let _ = child.wait();
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max).collect::<String>())
    }
}

fn resolve_config_path<'a>(
    config: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    let mut current = config;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn collect_config_keys(runs: &[Run]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for run in runs {
        if let Some(config) = &run.config {
            collect_keys_recursive(config, "", &mut keys);
        }
    }
    keys.into_iter().collect()
}

fn collect_keys_recursive(value: &serde_json::Value, prefix: &str, keys: &mut BTreeSet<String>) {
    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", prefix, key)
            };
            if val.is_object() {
                collect_keys_recursive(val, &path, keys);
            } else {
                keys.insert(path);
            }
        }
    }
}

fn collect_config_values(runs: &[Run], key: &str) -> Vec<String> {
    let mut values = BTreeSet::new();
    for run in runs {
        if let Some(config) = &run.config
            && let Some(val) = resolve_config_path(config, key)
        {
            values.insert(format_json_primitive(val));
        }
    }
    values.into_iter().collect()
}

fn run_matches_filters(run: &Run, filters: &[(String, String)]) -> bool {
    let config = match &run.config {
        Some(c) => c,
        None => return false,
    };
    filters.iter().all(|(key, val)| {
        resolve_config_path(config, key).is_some_and(|v| format_json_primitive(v) == *val)
    })
}

fn collect_differing_keys(configs: &[Option<&serde_json::Value>]) -> HashSet<String> {
    let mut diffs = HashSet::new();
    collect_diffs_recursive(configs, "", &mut diffs);
    diffs
}

fn collect_diffs_recursive(
    configs: &[Option<&serde_json::Value>],
    prefix: &str,
    diffs: &mut HashSet<String>,
) {
    let mut all_keys: Vec<String> = Vec::new();
    for cfg in configs.iter().flatten() {
        if let serde_json::Value::Object(map) = cfg {
            for key in map.keys() {
                if !all_keys.contains(key) {
                    all_keys.push(key.clone());
                }
            }
        }
    }

    for key in &all_keys {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{}.{}", prefix, key)
        };

        let values: Vec<Option<&serde_json::Value>> = configs
            .iter()
            .map(|cfg| cfg.and_then(|c| c.get(key)))
            .collect();

        let present: Vec<&serde_json::Value> = values.iter().copied().flatten().collect();

        if present.len() != configs.iter().filter(|c| c.is_some()).count() {
            diffs.insert(path.clone());
        } else if present.len() >= 2 {
            let all_objects = present.iter().all(|v| v.is_object());
            if all_objects {
                let sub_configs: Vec<Option<&serde_json::Value>> =
                    present.iter().map(|v| Some(*v)).collect();
                collect_diffs_recursive(&sub_configs, &path, diffs);
            } else if !present.windows(2).all(|w| w[0] == w[1]) {
                diffs.insert(path.clone());
            }
        }
    }
}

fn render_json_value(
    value: &serde_json::Value,
    indent: usize,
    lines: &mut Vec<Line>,
    diff_keys: Option<&HashSet<String>>,
    path_prefix: &str,
) {
    let pad = "  ".repeat(indent);
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let path = if path_prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path_prefix, key)
                };
                let is_diff = diff_keys.is_some_and(|d| d.contains(&path));
                match val {
                    serde_json::Value::Object(_) => {
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}:", key), Style::default().fg(NEON_MAGENTA)),
                        ]));
                        render_json_value(val, indent + 1, lines, diff_keys, &path);
                    }
                    serde_json::Value::Array(arr) => {
                        let val_style = if is_diff {
                            Style::default().fg(NEON_YELLOW).bold()
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                            Span::styled(format!("[{}]", arr.len()), val_style),
                        ]));
                    }
                    _ => {
                        let val_str = truncate_str(&format_json_primitive(val), 28);
                        let val_style = if is_diff {
                            Style::default().fg(NEON_YELLOW).bold()
                        } else {
                            Style::default().fg(Color::White)
                        };
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                            Span::styled(val_str, val_style),
                        ]));
                    }
                }
            }
        }
        _ => {
            let val_str = truncate_str(&format_json_primitive(value), 28);
            lines.push(Line::from(Span::styled(
                format!("{}{}", pad, val_str),
                Style::default().fg(Color::White),
            )));
        }
    }
}

fn format_json_primitive(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => format!("[{} items]", arr.len()),
        serde_json::Value::Object(_) => "{...}".to_string(),
    }
}

fn lttb_downsample(data: &[(f64, f64)], threshold: usize) -> Vec<(f64, f64)> {
    if data.len() <= threshold || threshold < 3 {
        return data.to_vec();
    }

    let mut sampled = Vec::with_capacity(threshold);
    sampled.push(data[0]);

    let bucket_size = (data.len() - 2) as f64 / (threshold - 2) as f64;

    let mut a_idx = 0usize;

    for i in 0..(threshold - 2) {
        let avg_start = ((i + 1) as f64 * bucket_size).floor() as usize + 1;
        let avg_end = (((i + 2) as f64 * bucket_size).floor() as usize + 1).min(data.len());

        let (avg_x, avg_y) = if avg_end > avg_start {
            let count = (avg_end - avg_start) as f64;
            let sum_x: f64 = data[avg_start..avg_end].iter().map(|p| p.0).sum();
            let sum_y: f64 = data[avg_start..avg_end].iter().map(|p| p.1).sum();
            (sum_x / count, sum_y / count)
        } else {
            data[avg_start.min(data.len() - 1)]
        };

        let range_start = (i as f64 * bucket_size).floor() as usize + 1;
        let range_end = ((i + 1) as f64 * bucket_size).floor() as usize + 1;

        let (ax, ay) = data[a_idx];
        let mut max_area = -1.0f64;
        let mut max_idx = range_start;

        for (j, point) in data
            .iter()
            .enumerate()
            .take(range_end.min(data.len()))
            .skip(range_start)
        {
            let area = ((point.0 - ax) * (avg_y - ay) - (avg_x - ax) * (point.1 - ay)).abs();
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }

        sampled.push(data[max_idx]);
        a_idx = max_idx;
    }

    sampled.push(data[data.len() - 1]);
    sampled
}

fn points_to_xy(points: &[MetricPoint], x_axis: XAxis) -> Vec<(f64, f64)> {
    match x_axis {
        XAxis::Step => points.iter().map(|p| (p.step as f64, p.value)).collect(),
        XAxis::Timestamp => {
            let first_ts = points.first().map(|p| p.timestamp).unwrap_or(0.0);
            points
                .iter()
                .map(|p| (p.timestamp - first_ts, p.value))
                .collect()
        }
    }
}

fn is_scalar_series(points: &[MetricPoint]) -> bool {
    match points.len() {
        0 | 1 => true,
        _ => {
            let s0 = points[0].step;
            points.iter().all(|p| p.step == s0)
        }
    }
}

fn format_scalar_value(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    if !(1e-3..1e6).contains(&abs) {
        format!("{:.4e}", v)
    } else {
        let s = format!("{:.4}", v);
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed.is_empty() || trimmed == "-" {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    }
}

fn format_x_label(value: f64, span: f64, x_axis: XAxis) -> String {
    match x_axis {
        XAxis::Step => format!("{:.0}", value),
        XAxis::Timestamp => format_duration_label(value, span),
    }
}

fn format_y_label(value: f64, range: f64) -> String {
    let scale = value.abs().max(range.abs());
    if scale == 0.0 {
        return "0".to_string();
    }
    if !(1e-3..1e6).contains(&scale) {
        return format!("{:.1e}", value);
    }
    let decimals = if range >= 100.0 {
        0
    } else if range >= 10.0 {
        1
    } else if range >= 1.0 {
        2
    } else if range >= 0.1 {
        3
    } else if range >= 0.01 {
        4
    } else {
        5
    };
    format!("{:.*}", decimals, value)
}

fn format_duration_label(seconds: f64, span: f64) -> String {
    let span = span.max(1e-9);
    let s = seconds.max(0.0);
    if span < 120.0 {
        format!("{}s", s.round() as i64)
    } else if span < 600.0 {
        let m = (s / 60.0).floor() as i64;
        let rem = (s - (m as f64) * 60.0).round() as i64;
        if rem == 0 {
            format!("{}m", m)
        } else {
            format!("{}m{}s", m, rem)
        }
    } else if span < 7200.0 {
        format!("{}m", (s / 60.0).round() as i64)
    } else if span < 172800.0 {
        let h = (s / 3600.0).floor() as i64;
        let rem_m = ((s - (h as f64) * 3600.0) / 60.0).round() as i64;
        if rem_m == 0 {
            format!("{}h", h)
        } else {
            format!("{}h{}m", h, rem_m)
        }
    } else {
        let d = (s / 86400.0).floor() as i64;
        let rem_h = ((s - (d as f64) * 86400.0) / 3600.0).round() as i64;
        if rem_h == 0 {
            format!("{}d", d)
        } else {
            format!("{}d{}h", d, rem_h)
        }
    }
}

fn render_chart(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    points: &[MetricPoint],
    selected: bool,
    x_axis: XAxis,
) {
    use ratatui::symbols::Marker;
    use ratatui::widgets::{Axis, Chart, Dataset, GraphType};

    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let display_title = match x_axis {
        XAxis::Step => title.to_string(),
        XAxis::Timestamp => format!("{} (time)", title),
    };

    if points.is_empty() {
        let block = Block::default()
            .title(Span::styled(display_title, title_style))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color));
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No data",
                Style::default().fg(Color::DarkGray),
            ))
            .block(block),
            area,
        );
        return;
    }

    if is_scalar_series(points) {
        render_scalar_value_card(frame, area, &display_title, points, selected);
        return;
    }

    let raw_data: Vec<(f64, f64)> = points_to_xy(points, x_axis);
    let max_points = (area.width as usize) * 2;
    let data = lttb_downsample(&raw_data, max_points);

    let x_min = raw_data.first().map(|p| p.0).unwrap_or(0.0);
    let x_max = raw_data.last().map(|p| p.0).unwrap_or(1.0);
    let y_data_min = raw_data.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_data_max = raw_data
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    // Pad chart bounds so data doesn't clip at edges, but use actual data range for labels
    let y_range = (y_data_max - y_data_min).max(0.001);
    let y_min = y_data_min - y_range * 0.1;
    let y_max = y_data_max + y_range * 0.1;

    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(NEON_GREEN))
        .data(&data);

    let axis_style = Style::default().fg(DIM_CYAN);
    let label_style = Style::default().fg(Color::DarkGray);

    let x_span = x_max - x_min;
    let num_x_ticks = 5;
    let x_labels: Vec<Span> = (0..num_x_ticks)
        .map(|i| {
            let v = x_min + x_span * i as f64 / (num_x_ticks - 1) as f64;
            Span::styled(format_x_label(v, x_span, x_axis), label_style)
        })
        .collect();

    let num_y_ticks = 5;
    let y_range = y_data_max - y_data_min;
    let y_labels: Vec<Span> = (0..num_y_ticks)
        .map(|i| {
            let v = y_data_min + y_range * i as f64 / (num_y_ticks - 1) as f64;
            Span::styled(format_y_label(v, y_range), label_style)
        })
        .collect();

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title(Span::styled(display_title, title_style))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color)),
        )
        .x_axis(
            Axis::default()
                .style(axis_style)
                .bounds([x_min, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .style(axis_style)
                .bounds([y_min, y_max])
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

fn render_scalar_value_card(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    points: &[MetricPoint],
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let block = Block::default()
        .title(Span::styled(title.to_string(), title_style))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(last) = points.last() else {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "No data",
                Style::default().fg(Color::DarkGray),
            )),
            inner,
        );
        return;
    };

    let value_str = format_scalar_value(last.value);
    let step_str = format!("step {}", last.step);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            value_str,
            Style::default().fg(NEON_GREEN).bold(),
        ))
        .alignment(Alignment::Center),
        Line::from(""),
        Line::from(Span::styled(step_str, Style::default().fg(Color::DarkGray)))
            .alignment(Alignment::Center),
    ];

    let pad_top = (inner.height as i32 - lines.len() as i32).max(0) / 2;
    let mut padded: Vec<Line> = (0..pad_top).map(|_| Line::from("")).collect();
    padded.extend(lines);

    frame.render_widget(Paragraph::new(padded), inner);
}

fn render_examples_card(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    group: &ExampleGroup,
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let content: Vec<Line> = if let Some(example) = group.last() {
        let max_lines = area.height.saturating_sub(4) as usize;
        let first_prompt = example.prompts.first().map(|s| s.as_str()).unwrap_or("");
        let prompt_preview: String = first_prompt.chars().take(50).collect();
        let first_response = example
            .responses
            .first()
            .and_then(|r| r.first())
            .map(|s| s.as_str())
            .unwrap_or("");
        let response_lines: Vec<&str> = first_response
            .lines()
            .take(max_lines.saturating_sub(2))
            .collect();

        let mut lines = vec![Line::from(vec![
            Span::styled("Q: ", Style::default().fg(NEON_YELLOW).bold()),
            Span::styled(
                format!("{}...", prompt_preview),
                Style::default().fg(Color::White),
            ),
        ])];
        if let Some(gt) = example.groundtruth.as_ref().and_then(|g| g.first()) {
            let gt_preview: String = gt.chars().take(50).collect();
            lines.push(Line::from(vec![
                Span::styled("GT: ", Style::default().fg(NEON_MAGENTA).bold()),
                Span::styled(
                    format!("{}...", gt_preview),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "A: ",
            Style::default().fg(NEON_GREEN).bold(),
        )));
        for line in response_lines {
            lines.push(Line::from(Span::styled(
                line,
                Style::default().fg(Color::Gray),
            )));
        }
        lines
    } else {
        vec![Line::from(Span::styled(
            "No examples",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    let avg_reward = group.avg_reward();

    let batch_size = group.last().map(|e| e.prompts.len()).unwrap_or(0);
    let mut title_spans = vec![Span::styled(format!("{} ", name), title_style)];

    if batch_size > 1 {
        title_spans.push(Span::styled(
            format!("({} total, batch={})", group.len(), batch_size),
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        title_spans.push(Span::styled(
            format!("({} total)", group.len()),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if let Some(avg) = avg_reward {
        let reward_color = reward_color(avg);
        title_spans.push(Span::styled(" ", Style::default()));
        title_spans.push(Span::styled(
            format!("[avg: {:.2}]", avg),
            Style::default().fg(reward_color),
        ));
    }

    let title = Line::from(title_spans);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let paragraph = Paragraph::new(content)
        .block(block)
        .wrap(ratatui::widgets::Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn format_size(size: u64) -> String {
    if size >= 1_073_741_824 {
        format!("{:.1} GB", size as f64 / 1_073_741_824.0)
    } else if size >= 1_048_576 {
        format!("{:.1} MB", size as f64 / 1_048_576.0)
    } else {
        format!("{:.1} KB", size as f64 / 1024.0)
    }
}

fn val_metrics_at_step(
    metrics: &HashMap<String, Vec<MetricPoint>>,
    step: u64,
) -> Vec<(String, f64)> {
    let mut results: Vec<(String, f64)> = metrics
        .iter()
        .filter(|(k, _)| k.starts_with("val"))
        .filter_map(|(k, points)| {
            let idx = points.partition_point(|p| p.step < step);
            if idx < points.len() && points[idx].step == step {
                let short = k
                    .strip_prefix("val/")
                    .or_else(|| k.strip_prefix("val_"))
                    .unwrap_or(k);
                Some((short.to_string(), points[idx].value))
            } else {
                None
            }
        })
        .collect();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

fn render_checkpoints_card(
    frame: &mut Frame,
    area: Rect,
    checkpoints: &[Checkpoint],
    metrics: &HashMap<String, Vec<MetricPoint>>,
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let max_lines = area.height.saturating_sub(3) as usize;
    let mut content: Vec<Line> = Vec::new();
    for ckpt in checkpoints {
        if content.len() >= max_lines {
            break;
        }
        let status_icon = if ckpt.all_downloaded() {
            "● "
        } else {
            "○ "
        };
        let status_color = if ckpt.all_downloaded() {
            NEON_GREEN
        } else {
            Color::DarkGray
        };
        let mut spans = vec![
            Span::styled(status_icon, Style::default().fg(status_color)),
            Span::styled("step ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", ckpt.step),
                Style::default().fg(NEON_YELLOW).bold(),
            ),
        ];
        if let Some(ts) = ckpt.timestamp {
            spans.push(Span::styled(
                format!("  {}", ts.format("%Y-%m-%d %H:%M")),
                Style::default().fg(Color::DarkGray),
            ));
        }
        if let Some(size) = ckpt.total_size_bytes() {
            spans.push(Span::styled(
                format!("  {}", format_size(size)),
                Style::default().fg(NEON_GREEN),
            ));
        }
        content.push(Line::from(spans));

        let val = val_metrics_at_step(metrics, ckpt.step);
        for (name, value) in &val {
            if content.len() >= max_lines {
                break;
            }
            content.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(format!("{}=", name), Style::default().fg(NEON_MAGENTA)),
                Span::styled(format!("{:.4}", value), Style::default().fg(Color::White)),
            ]));
        }
    }

    let title = Line::from(vec![
        Span::styled("Checkpoints ", title_style),
        Span::styled(
            format!("({})", checkpoints.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let paragraph = Paragraph::new(content).block(block);
    frame.render_widget(paragraph, area);
}

fn cm_cell_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = ((1.0 - t) * 0.0) as u8;
    let g = ((1.0 - t) * 40.0 + t * 255.0) as u8;
    let b = ((1.0 - t) * 60.0 + t * 136.0) as u8;
    Color::Rgb(r, g, b)
}

fn render_confusion_matrix_card(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    point: &ConfusionMatrixPoint,
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    let title = Line::from(vec![
        Span::styled(format!(" {} ", name), title_style),
        Span::styled(
            format!("@ step {} ", point.step),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let n = point.matrix.len();
    if n == 0 || inner.width == 0 || inner.height == 0 {
        return;
    }

    let max_val = point
        .matrix
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0)
        .max(1) as f64;

    let cell_w = ((inner.width as usize) / n).clamp(1, 4) as u16;
    let cell_h = ((inner.height as usize) / n).clamp(1, 2) as u16;

    let grid_w = cell_w * n as u16;
    let grid_h = cell_h * n as u16;
    let x0 = inner.x + (inner.width.saturating_sub(grid_w)) / 2;
    let y0 = inner.y + (inner.height.saturating_sub(grid_h)) / 2;

    let block_str: String = "█".repeat(cell_w as usize);

    for i in 0..n {
        for j in 0..n {
            let v = point.matrix[i][j] as f64;
            let t = v / max_val;
            let color = cm_cell_color(t);
            let style = Style::default().fg(color);
            for dy in 0..cell_h {
                let y = y0 + (i as u16) * cell_h + dy;
                let x = x0 + (j as u16) * cell_w;
                if y >= inner.bottom() || x >= inner.right() {
                    continue;
                }
                let line_area = Rect::new(x, y, cell_w.min(inner.right() - x), 1);
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(block_str.clone(), style))),
                    line_area,
                );
            }
        }
    }
}

fn render_focused_confusion_matrix(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    point: &ConfusionMatrixPoint,
    step_idx: usize,
    total_steps: usize,
) {
    let title = Line::from(vec![
        Span::styled(format!(" {} ", name), Style::default().fg(NEON_CYAN).bold()),
        Span::styled(
            format!("step {}  [{}/{}] ", point.step, step_idx + 1, total_steps),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(NEON_CYAN));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let n = point.matrix.len();
    if n == 0 || inner.width < 6 || inner.height < 4 {
        return;
    }

    let max_label_len = point
        .labels
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .min(12);

    let max_val = point.matrix.iter().flatten().copied().max().unwrap_or(0);
    let cell_text_w = max_val.to_string().len().max(3) + 1;

    let row_label_w = (max_label_len + 2) as u16;
    let header_h: u16 = 2;

    let available_w = inner.width.saturating_sub(row_label_w);
    let available_h = inner.height.saturating_sub(header_h);

    let cell_w = (available_w as usize / n).clamp(cell_text_w, cell_text_w.max(10)) as u16;
    let cell_h = (available_h as usize / n).clamp(1, 3) as u16;
    if cell_w == 0 || cell_h == 0 {
        return;
    }

    let grid_x = inner.x + row_label_w;
    let grid_y = inner.y + header_h;

    for (j, label) in point.labels.iter().enumerate().take(n) {
        let x = grid_x + (j as u16) * cell_w;
        if x >= inner.right() {
            break;
        }
        let width = cell_w.min(inner.right() - x);
        let truncated: String = label.chars().take(cell_w as usize).collect();
        let centered = format!("{:^w$}", truncated, w = cell_w as usize);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                centered,
                Style::default().fg(NEON_MAGENTA),
            ))),
            Rect::new(x, inner.y, width, 1),
        );
    }

    let max_val_f = (max_val as f64).max(1.0);
    for i in 0..n {
        let row_y = grid_y + (i as u16) * cell_h;
        if row_y >= inner.bottom() {
            break;
        }
        let label = point.labels.get(i).map(String::as_str).unwrap_or("");
        let truncated: String = label.chars().take(max_label_len).collect();
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("{:>w$} ", truncated, w = max_label_len),
                Style::default().fg(NEON_MAGENTA),
            ))),
            Rect::new(inner.x, row_y + cell_h / 2, row_label_w, 1),
        );

        for j in 0..n {
            let x = grid_x + (j as u16) * cell_w;
            if x >= inner.right() {
                break;
            }
            let width = cell_w.min(inner.right() - x);
            let v = point.matrix[i][j];
            let t = v as f64 / max_val_f;
            let bg = cm_cell_color(t);
            let fg = if t > 0.55 { Color::Black } else { Color::White };
            let text = format!("{:^w$}", v, w = cell_w as usize);
            let style = Style::default().fg(fg).bg(bg);
            let mid = row_y + cell_h / 2;
            for dy in 0..cell_h {
                let y = row_y + dy;
                if y >= inner.bottom() {
                    break;
                }
                let line_text = if y == mid {
                    text.clone()
                } else {
                    " ".repeat(cell_w as usize)
                };
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(line_text, style))),
                    Rect::new(x, y, width, 1),
                );
            }
        }
    }
}

fn render_focused_checkpoints(
    frame: &mut Frame,
    area: Rect,
    checkpoints: &[Checkpoint],
    metrics: &HashMap<String, Vec<MetricPoint>>,
    selected_index: usize,
) {
    let visible_height = area.height as usize;

    let line_counts: Vec<usize> = checkpoints
        .iter()
        .map(|ckpt| 1 + val_metrics_at_step(metrics, ckpt.step).len())
        .collect();

    let mut scroll_offset = 0;
    let mut cumulative = 0;
    for i in 0..checkpoints.len() {
        let needed = line_counts[i..=selected_index.min(checkpoints.len() - 1)]
            .iter()
            .sum::<usize>();
        if needed <= visible_height {
            scroll_offset = i;
            break;
        }
        cumulative += line_counts[i];
        scroll_offset = i + 1;
    }
    let _ = cumulative;

    let mut lines: Vec<Line> = Vec::new();
    for (i, ckpt) in checkpoints.iter().enumerate().skip(scroll_offset) {
        if lines.len() >= visible_height {
            break;
        }
        let is_selected = i == selected_index;
        let cursor = if is_selected { "▶ " } else { "  " };
        let status_icon = if ckpt.all_downloaded() {
            "● "
        } else {
            "○ "
        };
        let status_color = if ckpt.all_downloaded() {
            NEON_GREEN
        } else {
            Color::DarkGray
        };

        let mut spans = vec![
            Span::styled(
                cursor,
                Style::default().fg(if is_selected {
                    NEON_CYAN
                } else {
                    Color::DarkGray
                }),
            ),
            Span::styled(status_icon, Style::default().fg(status_color)),
            Span::styled("step ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", ckpt.step),
                Style::default().fg(NEON_YELLOW).bold(),
            ),
        ];
        if let Some(ts) = ckpt.timestamp {
            spans.push(Span::styled(
                format!("  {}", ts.format("%Y-%m-%d %H:%M")),
                Style::default().fg(Color::DarkGray),
            ));
        }
        if let Some(size) = ckpt.total_size_bytes() {
            spans.push(Span::styled(
                format!("  {}", format_size(size)),
                Style::default().fg(NEON_GREEN),
            ));
        }
        if !ckpt.files.is_empty() {
            let file_info: Vec<String> = ckpt
                .files
                .iter()
                .map(|f| {
                    let dl = if ckpt.downloaded_files.contains(&f.name) {
                        "●"
                    } else {
                        "○"
                    };
                    let sz = f
                        .size_bytes
                        .map(|s| format!(" {}", format_size(s)))
                        .unwrap_or_default();
                    format!("{} {}{}", dl, f.name, sz)
                })
                .collect();
            spans.push(Span::styled(
                format!("  [{}]", file_info.join(", ")),
                Style::default().fg(Color::DarkGray),
            ));
        }

        let bg = if is_selected {
            Color::Rgb(30, 30, 50)
        } else {
            Color::Reset
        };
        lines.push(Line::from(spans).style(Style::default().bg(bg)));

        let val = val_metrics_at_step(metrics, ckpt.step);
        for (name, value) in &val {
            if lines.len() >= visible_height {
                break;
            }
            lines.push(
                Line::from(vec![
                    Span::raw("      "),
                    Span::styled(format!("{}=", name), Style::default().fg(NEON_MAGENTA)),
                    Span::styled(format!("{:.4}", value), Style::default().fg(Color::White)),
                ])
                .style(Style::default().bg(bg)),
            );
        }
    }

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, area);
}

fn reward_color(reward: f64) -> Color {
    if reward >= 0.7 {
        NEON_GREEN
    } else if reward >= 0.3 {
        NEON_YELLOW
    } else {
        NEON_MAGENTA
    }
}

fn render_focused(app: &App, frame: &mut Frame) {
    let area = frame.area();

    if app.compare_focused {
        render_focused_compare(app, frame, area);
    } else {
        match app.view_mode {
            ViewMode::Runs => render_focused_run(app, frame, area),
            ViewMode::Artifacts | ViewMode::Infra => {}
        }
    }

    if let Some(ref input) = app.goto_step_input {
        use ratatui::widgets::Clear;
        let popup_width = 28u16.min(area.width.saturating_sub(4));
        let popup_height = 3;
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);
        frame.render_widget(Clear, popup_area);
        let text = format!("Go to step: {}_", input);
        let popup = Paragraph::new(text)
            .style(Style::default().fg(NEON_CYAN))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(NEON_MAGENTA))
                    .border_type(BorderType::Rounded)
                    .title("goto"),
            );
        frame.render_widget(popup, popup_area);
    }
}

fn render_focused_compare(app: &App, frame: &mut Frame, area: Rect) {
    let cards = app.compare_cards();
    let Some(card) = cards.get(app.selected_card) else {
        frame.render_widget(Paragraph::new("No card selected"), area);
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    match card {
        Card::Chart { name } => {
            let x_axis = app.chart_x_axis(name);
            let run_data: Vec<CompareRunData> = app
                .compared_runs
                .iter()
                .enumerate()
                .filter_map(|(ci, &run_idx)| {
                    let run = app.runs.get(run_idx)?;
                    let color = COMPARE_COLORS[ci % COMPARE_COLORS.len()];
                    let pts = run.metrics.get(name);
                    let is_scalar = pts.map(|p| is_scalar_series(p)).unwrap_or(true);
                    let data: Vec<(f64, f64)> =
                        pts.map(|p| points_to_xy(p, x_axis)).unwrap_or_default();
                    Some((run.display_name(), color, data, is_scalar))
                })
                .collect();
            render_comparison_chart(frame, chunks[0], name, &run_data, true, x_axis);
        }
        Card::Examples { name } => {
            render_focused_compare_examples(app, frame, chunks[0], name);
        }
        _ => {}
    }

    let mut footer_spans = vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("←→", Style::default().fg(NEON_CYAN)),
        Span::styled("] card ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}/{}", app.selected_card + 1, cards.len()),
            Style::default().fg(NEON_GREEN),
        ),
    ];

    if let Card::Examples { name } = card {
        let steps = app.compare_example_steps(name);
        if !steps.is_empty() {
            footer_spans.extend(vec![
                Span::styled("  [", Style::default().fg(DIM_CYAN)),
                Span::styled("↑↓/⇧", Style::default().fg(NEON_CYAN)),
                Span::styled("] step ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/{}", app.selected_example + 1, steps.len()),
                    Style::default().fg(NEON_GREEN),
                ),
            ]);
        }
    }

    footer_spans.extend(vec![
        Span::styled("  [", Style::default().fg(DIM_CYAN)),
        Span::styled("?", Style::default().fg(NEON_YELLOW)),
        Span::styled("] help", Style::default().fg(Color::DarkGray)),
    ]);

    let footer = Line::from(footer_spans);
    frame.render_widget(Paragraph::new(footer), chunks[1]);
}

fn render_focused_compare_examples(app: &App, frame: &mut Frame, area: Rect, name: &str) {
    let steps = app.compare_example_steps(name);
    if steps.is_empty() {
        frame.render_widget(Paragraph::new("No examples available"), area);
        return;
    }

    let step_idx = app.selected_example.min(steps.len().saturating_sub(1));
    let current_step = steps[step_idx];

    let header_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3)])
        .split(area);

    let header = Line::from(vec![
        Span::styled("◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled(name, Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" │ ", Style::default().fg(DIM_CYAN)),
        Span::styled("step ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", current_step), Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("  ({}/{})", step_idx + 1, steps.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), header_chunks[0]);

    let num_runs = app.compared_runs.len();
    let constraints: Vec<Constraint> = (0..num_runs)
        .map(|_| Constraint::Ratio(1, num_runs as u32))
        .collect();
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(header_chunks[1]);

    for (ci, &run_idx) in app.compared_runs.iter().enumerate() {
        let Some(run) = app.runs.get(run_idx) else {
            continue;
        };
        let color = COMPARE_COLORS[ci % COMPARE_COLORS.len()];
        let col_area = columns[ci];

        let example = run
            .examples
            .get(name)
            .and_then(|g| g.find_step(current_step))
            .and_then(|idx| run.examples.get(name).unwrap().load(idx));

        if let Some(ex) = example {
            let has_gt = ex.groundtruth.is_some();
            let col_chunks = if has_gt {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(30),
                        Constraint::Percentage(35),
                        Constraint::Percentage(35),
                    ])
                    .split(col_area)
            } else {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                    .split(col_area)
            };

            let prompt_text = ex
                .prompts
                .get(app.selected_prompt)
                .cloned()
                .unwrap_or_default();

            let prompt_block = Block::default()
                .title(Line::from(vec![
                    Span::styled("● ", Style::default().fg(color)),
                    Span::styled(
                        format!("{} ", run.display_name()),
                        Style::default().fg(color),
                    ),
                    Span::styled("PROMPT", Style::default().fg(NEON_YELLOW).bold()),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(color));

            let prompt = Paragraph::new(prompt_text)
                .style(Style::default().fg(Color::White))
                .block(prompt_block)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .scroll((app.prompt_scroll_offset as u16, 0));
            frame.render_widget(prompt, col_chunks[0]);

            let response_chunk_idx = if has_gt {
                let gt_text = ex
                    .groundtruth
                    .as_ref()
                    .and_then(|gt| gt.get(app.selected_prompt))
                    .cloned()
                    .unwrap_or_default();
                let gt_block = Block::default()
                    .title(Span::styled(
                        "GROUNDTRUTH",
                        Style::default().fg(NEON_MAGENTA).bold(),
                    ))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(color));
                let gt = Paragraph::new(gt_text)
                    .style(Style::default().fg(Color::White))
                    .block(gt_block)
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .scroll((app.groundtruth_scroll_offset as u16, 0));
                frame.render_widget(gt, col_chunks[1]);
                2
            } else {
                1
            };

            let response_text = ex
                .responses
                .get(app.selected_prompt)
                .and_then(|r| r.get(app.selected_response))
                .cloned()
                .unwrap_or_default();

            let mut response_title_spans = vec![Span::styled(
                "RESPONSE",
                Style::default().fg(NEON_GREEN).bold(),
            )];

            let reward = ex
                .rewards
                .as_ref()
                .and_then(|rewards| rewards.get(app.selected_prompt))
                .and_then(|prompt_rewards| prompt_rewards.get(app.selected_response));

            if let Some(reward) = reward {
                response_title_spans.push(Span::styled(" │ ", Style::default().fg(DIM_CYAN)));
                match reward {
                    Reward::Scalar(v) => {
                        let rc = reward_color(*v);
                        response_title_spans.push(Span::styled(
                            format!("reward: {:.2}", v),
                            Style::default().fg(rc),
                        ));
                    }
                    Reward::Components(map) => {
                        let mut parts: Vec<(&String, &f64)> = map.iter().collect();
                        parts.sort_by_key(|(k, _)| *k);
                        for (i, (key, value)) in parts.iter().enumerate() {
                            let rc = reward_color(**value);
                            if i > 0 {
                                response_title_spans.push(Span::styled("  ", Style::default()));
                            }
                            response_title_spans.push(Span::styled(
                                format!("{}: {:.2}", key, value),
                                Style::default().fg(rc),
                            ));
                        }
                    }
                }
            }

            let response_block = Block::default()
                .title(Line::from(response_title_spans))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(color));

            let response = Paragraph::new(response_text)
                .style(Style::default().fg(Color::Gray))
                .block(response_block)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .scroll((app.response_scroll_offset as u16, 0));
            frame.render_widget(response, col_chunks[response_chunk_idx]);
        } else {
            let block = Block::default()
                .title(Line::from(vec![
                    Span::styled("● ", Style::default().fg(color)),
                    Span::styled(run.display_name(), Style::default().fg(color)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(color));

            let msg = Paragraph::new(Span::styled(
                format!("No example at step {}", current_step),
                Style::default().fg(Color::DarkGray),
            ))
            .block(block);
            frame.render_widget(msg, col_area);
        }
    }
}

fn render_focused_run(app: &App, frame: &mut Frame, area: Rect) {
    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    let cards = app.cards();
    let Some(card) = cards.get(app.selected_card) else {
        frame.render_widget(Paragraph::new("No card selected"), area);
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    match card {
        Card::Chart { name } => {
            if let Some(points) = run.metrics.get(name) {
                let x_axis = app.chart_x_axis(name);
                render_chart(frame, chunks[0], name, points, true, x_axis);
            }
            let footer = Line::from(vec![
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("←→", Style::default().fg(NEON_CYAN)),
                Span::styled("] card ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/{}  ", app.selected_card + 1, cards.len()),
                    Style::default().fg(NEON_GREEN),
                ),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("?", Style::default().fg(NEON_YELLOW)),
                Span::styled("] help", Style::default().fg(Color::DarkGray)),
            ]);
            frame.render_widget(Paragraph::new(footer), chunks[1]);
        }
        Card::Examples { name } => {
            if let Some(group) = run.examples.get(name) {
                let example = app
                    .cached_example
                    .as_ref()
                    .filter(|(n, i, _)| n == name && *i == app.selected_example)
                    .map(|(_, _, ex)| ex);
                if let Some(example) = example {
                    render_focused_example(
                        frame,
                        chunks[0],
                        name,
                        app.selected_example,
                        group.len(),
                        example,
                        app.selected_prompt,
                        app.selected_response,
                        app.focused_section,
                        app.prompt_scroll_offset,
                        app.response_scroll_offset,
                        app.groundtruth_scroll_offset,
                    );
                }
                let prompt_count = example.map(|e| e.prompts.len()).unwrap_or(0);
                let response_count = example
                    .and_then(|e| e.responses.get(app.selected_prompt))
                    .map(|r| r.len())
                    .unwrap_or(0);
                let focus_label = match app.focused_section {
                    FocusedSection::Prompt => "prompt",
                    FocusedSection::Groundtruth => "groundtruth",
                    FocusedSection::Response => "response",
                };
                let mut footer_spans = vec![
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                    Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("←→", Style::default().fg(NEON_CYAN)),
                    Span::styled("] card ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", app.selected_card + 1, cards.len()),
                        Style::default().fg(NEON_GREEN),
                    ),
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("Tab", Style::default().fg(NEON_CYAN)),
                    Span::styled("] focus:", Style::default().fg(Color::DarkGray)),
                    Span::styled(focus_label, Style::default().fg(NEON_GREEN)),
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("↑↓/⇧", Style::default().fg(NEON_CYAN)),
                    Span::styled("] example ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", app.selected_example + 1, group.len()),
                        Style::default().fg(NEON_YELLOW),
                    ),
                ];
                if prompt_count > 1 {
                    footer_spans.extend(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled("[", Style::default().fg(DIM_CYAN)),
                        Span::styled("[]", Style::default().fg(NEON_CYAN)),
                        Span::styled("] prompt ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{}/{}", app.selected_prompt + 1, prompt_count),
                            Style::default().fg(NEON_YELLOW),
                        ),
                    ]);
                }
                if response_count > 1 {
                    footer_spans.extend(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled("[", Style::default().fg(DIM_CYAN)),
                        Span::styled("<>", Style::default().fg(NEON_CYAN)),
                        Span::styled("] response ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{}/{}", app.selected_response + 1, response_count),
                            Style::default().fg(NEON_MAGENTA),
                        ),
                    ]);
                }
                footer_spans.extend(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("?", Style::default().fg(NEON_YELLOW)),
                    Span::styled("] help", Style::default().fg(Color::DarkGray)),
                ]);
                let footer = Line::from(footer_spans);
                frame.render_widget(Paragraph::new(footer), chunks[1]);
            }
        }
        Card::ConfusionMatrix { name } => {
            if let Some(series) = run.confusion_matrices.get(name) {
                let idx = app
                    .confusion_step_idx
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| series.len().saturating_sub(1));
                let idx = idx.min(series.len().saturating_sub(1));
                if let Some(point) = series.get(idx) {
                    render_focused_confusion_matrix(
                        frame,
                        chunks[0],
                        name,
                        point,
                        idx,
                        series.len(),
                    );
                }
                let footer = Line::from(vec![
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                    Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("←→", Style::default().fg(NEON_CYAN)),
                    Span::styled("] card ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}  ", app.selected_card + 1, cards.len()),
                        Style::default().fg(NEON_GREEN),
                    ),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
                    Span::styled("] step ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}  ", idx + 1, series.len()),
                        Style::default().fg(NEON_YELLOW),
                    ),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("?", Style::default().fg(NEON_YELLOW)),
                    Span::styled("] help", Style::default().fg(Color::DarkGray)),
                ]);
                frame.render_widget(Paragraph::new(footer), chunks[1]);
            }
        }
        Card::Checkpoints => {
            if let Some(run) = app.current_run() {
                render_focused_checkpoints(
                    frame,
                    chunks[0],
                    &run.checkpoints,
                    &run.metrics,
                    app.selected_checkpoint,
                );
            }
            let footer = Line::from(vec![
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("←→", Style::default().fg(NEON_CYAN)),
                Span::styled("] card ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/{}  ", app.selected_card + 1, cards.len()),
                    Style::default().fg(NEON_GREEN),
                ),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("p", Style::default().fg(NEON_CYAN)),
                Span::styled("] download  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("?", Style::default().fg(NEON_YELLOW)),
                Span::styled("] help", Style::default().fg(Color::DarkGray)),
            ]);
            frame.render_widget(Paragraph::new(footer), chunks[1]);
        }
        Card::Artifacts => {
            if let Some(run) = app.current_run() {
                let run_artifacts = app.run_artifacts(run);
                render_focused_artifacts(frame, chunks[0], &run_artifacts);
            }
            let footer = Line::from(vec![
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("←→", Style::default().fg(NEON_CYAN)),
                Span::styled("] card ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/{}  ", app.selected_card + 1, cards.len()),
                    Style::default().fg(NEON_GREEN),
                ),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("?", Style::default().fg(NEON_YELLOW)),
                Span::styled("] help", Style::default().fg(Color::DarkGray)),
            ]);
            frame.render_widget(Paragraph::new(footer), chunks[1]);
        }
    }
}

fn render_focused_artifacts(frame: &mut Frame, area: Rect, artifacts: &[&Artifact]) {
    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
            Span::styled(
                format!("Artifacts ({})", artifacts.len()),
                Style::default().fg(NEON_CYAN).bold(),
            ),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    for artifact in artifacts {
        let icon = if artifact.content_type == "directory" {
            "📁"
        } else {
            "📄"
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", icon), Style::default()),
            Span::styled(&artifact.name, Style::default().fg(NEON_CYAN).bold()),
            Span::styled(
                format!("  ({})", artifact.display_size()),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        if !artifact.description.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(&artifact.description, Style::default().fg(Color::White)),
            ]));
        }
        lines.push(Line::from(""));
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

#[allow(clippy::too_many_arguments)]
fn render_focused_example(
    frame: &mut Frame,
    area: Rect,
    group_name: &str,
    index: usize,
    total: usize,
    example: &Example,
    selected_prompt: usize,
    selected_response: usize,
    focused_section: FocusedSection,
    prompt_scroll_offset: usize,
    response_scroll_offset: usize,
    groundtruth_scroll_offset: usize,
) {
    let has_gt = example.groundtruth.is_some();
    let chunks = if has_gt {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(35),
                Constraint::Percentage(35),
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area)
    };

    // Prompt title - show batch position if multiple prompts
    let prompt_focused = focused_section == FocusedSection::Prompt;
    let prompt_border_color = if prompt_focused { NEON_CYAN } else { DIM_CYAN };
    let mut prompt_title_spans = vec![Span::styled(
        "◆ PROMPT ",
        Style::default().fg(NEON_YELLOW).bold(),
    )];
    if example.prompts.len() > 1 {
        prompt_title_spans.extend(vec![
            Span::styled(
                format!("{}/{}", selected_prompt + 1, example.prompts.len()),
                Style::default().fg(NEON_YELLOW),
            ),
            Span::styled(" ", Style::default()),
        ]);
    }
    prompt_title_spans.extend(vec![
        Span::styled("│ ", Style::default().fg(DIM_CYAN)),
        Span::styled(group_name, Style::default().fg(NEON_MAGENTA)),
        Span::styled(" step ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", example.step), Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("  ({}/{})", index + 1, total),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let prompt_title = Line::from(prompt_title_spans);

    let prompt_text = example
        .prompts
        .get(selected_prompt)
        .cloned()
        .unwrap_or_default();
    let prompt = Paragraph::new(prompt_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(prompt_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(prompt_border_color)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false })
        .scroll((prompt_scroll_offset as u16, 0));
    frame.render_widget(prompt, chunks[0]);

    let response_chunk_idx = if has_gt {
        // Render groundtruth panel
        let gt_text = example
            .groundtruth
            .as_ref()
            .and_then(|gt| gt.get(selected_prompt))
            .cloned()
            .unwrap_or_default();
        let gt_focused = focused_section == FocusedSection::Groundtruth;
        let gt_border_color = if gt_focused { NEON_CYAN } else { DIM_CYAN };
        let gt_block = Block::default()
            .title(Span::styled(
                "◆ GROUNDTRUTH ",
                Style::default().fg(NEON_MAGENTA).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(gt_border_color));
        let gt = Paragraph::new(gt_text)
            .style(Style::default().fg(Color::White))
            .block(gt_block)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .scroll((groundtruth_scroll_offset as u16, 0));
        frame.render_widget(gt, chunks[1]);
        2
    } else {
        1
    };

    // Get responses for the selected prompt
    let responses_for_prompt = example
        .responses
        .get(selected_prompt)
        .cloned()
        .unwrap_or_default();

    // Get the reward for the selected response variant (if it exists)
    let current_reward = example
        .rewards
        .as_ref()
        .and_then(|rewards| rewards.get(selected_prompt))
        .and_then(|prompt_rewards| prompt_rewards.get(selected_response));

    // Response title with variant indicator if multiple responses
    let response_focused = focused_section == FocusedSection::Response;
    let response_border_color = if response_focused {
        NEON_CYAN
    } else {
        DIM_CYAN
    };

    let mut response_title_spans = vec![Span::styled(
        "◆ RESPONSE ",
        Style::default().fg(NEON_GREEN).bold(),
    )];

    if responses_for_prompt.len() > 1 {
        response_title_spans.extend(vec![
            Span::styled("│ ", Style::default().fg(DIM_CYAN)),
            Span::styled("group ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", selected_response + 1, responses_for_prompt.len()),
                Style::default().fg(NEON_MAGENTA),
            ),
        ]);
    }

    if let Some(reward) = current_reward {
        response_title_spans.push(Span::styled(" │ ", Style::default().fg(DIM_CYAN)));
        match reward {
            Reward::Scalar(v) => {
                let color = reward_color(*v);
                response_title_spans.push(Span::styled(
                    "reward: ",
                    Style::default().fg(Color::DarkGray),
                ));
                response_title_spans.push(Span::styled(
                    format!("{:.2}", v),
                    Style::default().fg(color),
                ));
            }
            Reward::Components(map) => {
                let mut parts: Vec<(&String, &f64)> = map.iter().collect();
                parts.sort_by_key(|(k, _)| *k);
                for (i, (key, value)) in parts.iter().enumerate() {
                    let color = reward_color(**value);
                    if i > 0 {
                        response_title_spans.push(Span::styled("  ", Style::default()));
                    }
                    response_title_spans.push(Span::styled(
                        format!("{}: ", key),
                        Style::default().fg(Color::DarkGray),
                    ));
                    response_title_spans.push(Span::styled(
                        format!("{:.2}", value),
                        Style::default().fg(color),
                    ));
                }
            }
        }
    }

    let response_title = Line::from(response_title_spans);

    // Get the selected response text
    let response_text = responses_for_prompt
        .get(selected_response)
        .cloned()
        .unwrap_or_default();

    let response = Paragraph::new(response_text)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .title(response_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(response_border_color)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false })
        .scroll((response_scroll_offset as u16, 0));
    frame.render_widget(response, chunks[response_chunk_idx]);
}

fn render_delete_confirm(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let (item_type, item_name) = if let Some(idx) = app.pending_delete_run {
        let name = app
            .runs
            .get(idx)
            .map(|r| r.name.as_str())
            .unwrap_or("unknown");
        ("run", name)
    } else if app.pending_delete_artifact {
        let name = app
            .artifacts
            .get(app.selected_artifact)
            .map(|a| a.name.as_str())
            .unwrap_or("unknown");
        ("artifact", name)
    } else if app.pending_delete_checkpoint {
        let step_str = app
            .current_run()
            .and_then(|r| r.checkpoints.get(app.selected_checkpoint))
            .map(|c| c.step.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let label = format!("step {}", step_str);
        let area = frame.area();
        let popup_width = 60u16.min(area.width.saturating_sub(4));
        let popup_height = 7u16;
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        let text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Delete checkpoint ", Style::default().fg(Color::White)),
                Span::styled(&label, Style::default().fg(NEON_YELLOW).bold()),
                Span::styled("? (y/n)", Style::default().fg(Color::White)),
            ]),
        ];
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(NEON_MAGENTA))
            .title(Span::styled(
                " Confirm Delete ",
                Style::default().fg(NEON_MAGENTA).bold(),
            ));
        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(Clear, popup_area);
        frame.render_widget(paragraph, popup_area);
        return;
    } else {
        ("item", "unknown")
    };

    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let popup_height = 7u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("Delete {} ", item_type),
                Style::default().fg(Color::White),
            ),
            Span::styled(item_name, Style::default().fg(NEON_CYAN).bold()),
            Span::styled("? (y/n)", Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "This action cannot be undone.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("y", Style::default().fg(NEON_GREEN)),
            Span::styled("] Yes  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("n", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] No  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " ⚠ CONFIRM DELETE ",
            Style::default().fg(NEON_MAGENTA).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_MAGENTA))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, popup_area);
}

fn render_pull_run_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 50u16.min(area.width.saturating_sub(4));
    let popup_height = 5u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_width = popup_width.saturating_sub(2) as usize;
    let input = &app.pull_run_input;
    let display = if input.len() >= inner_width {
        &input[input.len() - inner_width + 1..]
    } else {
        input.as_str()
    };
    let cursor = "_";

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(display, Style::default().fg(NEON_CYAN)),
            Span::styled(cursor, Style::default().fg(NEON_CYAN)),
        ]),
        Line::from(""),
    ];

    let block = Block::default()
        .title(Span::styled(
            " Pull Run (project/run) ",
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text).block(block);

    frame.render_widget(paragraph, popup_area);
}

fn render_checkpoint_download_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 40u16.min(area.width.saturating_sub(4));
    let popup_height =
        (app.checkpoint_download_options.len() as u16 + 4).min(area.height.saturating_sub(2));
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));
    for (i, opt) in app.checkpoint_download_options.iter().enumerate() {
        let is_sel = i == app.checkpoint_download_selected;
        let cursor = if is_sel { "▶ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(
                cursor,
                Style::default().fg(if is_sel { NEON_CYAN } else { Color::DarkGray }),
            ),
            Span::styled(
                opt.clone(),
                Style::default().fg(if is_sel { Color::White } else { Color::Gray }),
            ),
        ]));
    }

    let block = Block::default()
        .title(Span::styled(
            format!(" Download step {} ", app.checkpoint_download_step),
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn render_note_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let popup_height = 5u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_width = popup_width.saturating_sub(2) as usize;
    let input = &app.note_modal_input;
    let display = if input.len() >= inner_width {
        &input[input.len() - inner_width + 1..]
    } else {
        input.as_str()
    };
    let cursor = "_";

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(display, Style::default().fg(NEON_CYAN)),
            Span::styled(cursor, Style::default().fg(NEON_CYAN)),
        ]),
        Line::from(""),
    ];

    let block = Block::default()
        .title(Span::styled(
            " Note ",
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text).block(block);

    frame.render_widget(paragraph, popup_area);
}

fn render_move_run_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 50u16.min(area.width.saturating_sub(4));
    let popup_height = 7u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_width = popup_width.saturating_sub(2) as usize;
    let input = &app.move_run_input;
    let display = if input.len() >= inner_width {
        &input[input.len() - inner_width + 1..]
    } else {
        input.as_str()
    };
    let cursor = "_";

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(display, Style::default().fg(NEON_CYAN)),
            Span::styled(cursor, Style::default().fg(NEON_CYAN)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Leave empty to remove from project",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];

    let block = Block::default()
        .title(Span::styled(
            " Move to Project ",
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text).block(block);

    frame.render_widget(paragraph, popup_area);
}

fn render_filter_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let max_list_items = 15usize;

    let (items, title_str) = match app.filter_modal_phase {
        FilterPhase::KeySelect => {
            let keys = {
                let all = collect_config_keys(&app.runs);
                let query = app.filter_modal_input.to_lowercase();
                if query.is_empty() {
                    all
                } else {
                    all.into_iter()
                        .filter(|k| k.to_lowercase().contains(&query))
                        .collect()
                }
            };
            (keys, " Filter by Config ".to_string())
        }
        FilterPhase::ValueSelect => {
            let values = {
                let all = collect_config_values(&app.runs, &app.filter_modal_key);
                let query = app.filter_modal_input.to_lowercase();
                if query.is_empty() {
                    all
                } else {
                    all.into_iter()
                        .filter(|v| v.to_lowercase().contains(&query))
                        .collect()
                }
            };
            (values, format!(" Filter: {} = ? ", app.filter_modal_key))
        }
    };

    let filter_lines: Vec<Line> =
        if app.filter_modal_phase == FilterPhase::KeySelect && !app.config_filters.is_empty() {
            app.config_filters
                .iter()
                .enumerate()
                .map(|(i, (k, v))| {
                    let selected = i == app.filter_modal_selected;
                    let style = if selected {
                        Style::default().fg(NEON_YELLOW).bg(Color::Rgb(30, 40, 50))
                    } else {
                        Style::default().fg(NEON_YELLOW)
                    };
                    Line::from(vec![Span::styled(format!("  ✕ {}={}", k, v), style)])
                })
                .collect()
        } else {
            Vec::new()
        };

    let visible_items = items.len().min(max_list_items);
    let content_height = 2 + filter_lines.len() + visible_items + 1;
    let popup_height = (content_height as u16 + 2)
        .min(area.height.saturating_sub(4))
        .max(5);
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let inner_width = popup_width.saturating_sub(4) as usize;
    let input = &app.filter_modal_input;
    let display = if input.len() >= inner_width {
        &input[input.len() - inner_width + 1..]
    } else {
        input.as_str()
    };

    let mut text: Vec<Line> = Vec::new();
    text.push(Line::from(vec![
        Span::styled("  > ", Style::default().fg(NEON_MAGENTA)),
        Span::styled(display, Style::default().fg(NEON_CYAN)),
        Span::styled("_", Style::default().fg(NEON_CYAN)),
    ]));
    text.push(Line::from(""));

    text.extend(filter_lines);

    let offset = if app.filter_modal_phase == FilterPhase::KeySelect {
        app.config_filters.len()
    } else {
        0
    };

    for (i, item) in items.iter().enumerate().take(max_list_items) {
        let idx = offset + i;
        let selected = idx == app.filter_modal_selected;
        let style = if selected {
            Style::default().fg(NEON_CYAN).bg(Color::Rgb(30, 40, 50))
        } else {
            Style::default().fg(Color::Gray)
        };
        let prefix = if selected { "  ▸ " } else { "    " };
        text.push(Line::from(Span::styled(
            format!("{}{}", prefix, item),
            style,
        )));
    }

    let footer = match app.filter_modal_phase {
        FilterPhase::KeySelect => "[Enter] select  [Esc] close",
        FilterPhase::ValueSelect => "[Enter] select  [Esc] back",
    };
    text.push(Line::from(""));
    text.push(Line::from(Span::styled(
        format!("  {}", footer),
        Style::default().fg(Color::DarkGray),
    )));

    let block = Block::default()
        .title(Span::styled(
            title_str,
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn help_sections_for(app: &App) -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    let global: Vec<(&str, &str)> = vec![("?", "toggle this help"), ("Ctrl-C", "quit")];

    let sections: Vec<(&str, Vec<(&str, &str)>)> = match (app.view, app.view_mode) {
        (View::List, ViewMode::Runs) => vec![
            (
                "Navigation",
                vec![
                    ("↑↓", "move selection"),
                    ("Tab", "expand/collapse project"),
                    ("Enter", "open run"),
                    ("a / i", "artifacts / infra"),
                ],
            ),
            (
                "Compare",
                vec![
                    ("Space", "toggle run in comparison"),
                    ("v", "view comparison"),
                    ("Esc", "clear comparison"),
                ],
            ),
            (
                "Run actions",
                vec![
                    ("p", "pull selected run"),
                    ("P", "pull by name"),
                    ("d", "delete run"),
                    ("M", "move to project"),
                    ("s", "star / unstar"),
                    ("n", "edit note"),
                    ("A", "archive project"),
                ],
            ),
            (
                "Filter & search",
                vec![
                    ("/", "search"),
                    ("f", "add config filter"),
                    ("F", "clear filters"),
                    ("H", "toggle running-only"),
                    (".", "toggle archived visible"),
                ],
            ),
        ],
        (View::List, ViewMode::Artifacts) => vec![
            (
                "Navigation",
                vec![
                    ("↑↓", "move selection"),
                    ("Enter", "open artifact"),
                    ("r / i", "runs / infra"),
                ],
            ),
            (
                "Actions",
                vec![("p", "pull"), ("d", "download"), ("D", "delete")],
            ),
        ],
        (View::List, ViewMode::Infra) | (View::InfraList, _) => vec![
            (
                "Navigation",
                vec![
                    ("↑↓", "move selection"),
                    ("Tab", "switch panel (Instances/Types)"),
                    ("1-4", "switch provider"),
                    ("r / a", "runs / artifacts"),
                ],
            ),
            (
                "Instances",
                vec![
                    ("Enter", "SSH into instance"),
                    ("S", "setup + SSH"),
                    ("a", "add local machine"),
                    ("x", "terminate / remove"),
                    ("R", "refresh"),
                ],
            ),
            (
                "Types",
                vec![("Enter", "launch instance"), ("s", "cycle sort")],
            ),
        ],
        (View::RunDetail, _) => vec![
            (
                "Navigation",
                vec![
                    ("q / Esc", "back to list"),
                    ("←→", "move card selection"),
                    ("↑↓", "scroll cards grid"),
                    ("Enter", "focus selected card"),
                ],
            ),
            (
                "Actions",
                vec![
                    ("p", "pull run"),
                    ("d", "delete run"),
                    ("M", "move to project"),
                    ("n", "edit note"),
                    ("c", "toggle config panel"),
                    ("C", "full config view"),
                    ("S", "toggle system/train metrics"),
                    ("t", "toggle time/step x-axis"),
                ],
            ),
        ],
        (View::Focused, _) => vec![
            (
                "Navigation",
                vec![
                    ("q / Esc", "back"),
                    ("Tab", "cycle focused section"),
                    ("↑↓", "scroll section"),
                    ("[ ]", "previous / next prompt"),
                    ("< >", "previous / next response"),
                ],
            ),
            (
                "Actions",
                vec![
                    ("y", "copy focused section"),
                    ("g", "goto step"),
                    ("d", "download checkpoint"),
                ],
            ),
        ],
        (View::Compare, _) => vec![(
            "Navigation",
            vec![
                ("q / Esc", "back to runs list"),
                ("←→", "move card selection"),
                ("↑↓", "scroll cards grid"),
                ("Enter", "focus card"),
            ],
        )],
        (View::ConfigFull, _) => vec![
            (
                "Navigation",
                vec![("q / Esc", "back"), ("↑↓", "scroll / move")],
            ),
            ("Actions", vec![("y", "copy value"), ("Y", "copy all")]),
        ],
        (View::ArtifactDetail, _) => vec![
            ("Navigation", vec![("Esc / q", "back"), ("↑↓", "scroll")]),
            ("Actions", vec![("d", "download"), ("D", "delete")]),
        ],
        (View::InfraConfig, _) => vec![
            (
                "Navigation",
                vec![("↑↓", "move selection"), ("Esc / q", "back")],
            ),
            (
                "Actions",
                vec![("Enter / e", "edit key"), ("d", "delete key")],
            ),
        ],
        (View::S3Config, _) => vec![
            (
                "Navigation",
                vec![("↑↓", "move field"), ("Esc / q", "back")],
            ),
            (
                "Actions",
                vec![("Enter / e", "edit field"), ("d", "clear field")],
            ),
        ],
    };

    let mut all = sections;
    all.push(("Global", global));
    all
}

fn render_help_overlay(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let sections = help_sections_for(app);

    let mut lines: Vec<Line> = Vec::new();
    for (title, entries) in &sections {
        lines.push(Line::from(vec![Span::styled(
            format!(" {} ", title),
            Style::default().fg(NEON_MAGENTA).bold(),
        )]));
        for (key, desc) in entries {
            lines.push(Line::from(vec![
                Span::styled(format!("  {:>12}  ", key), Style::default().fg(NEON_CYAN)),
                Span::styled((*desc).to_string(), Style::default().fg(Color::Gray)),
            ]));
        }
        lines.push(Line::from(""));
    }
    lines.push(Line::from(vec![Span::styled(
        "  [Esc/q/?] close   [↑↓/PgUp/PgDn] scroll",
        Style::default().fg(Color::DarkGray),
    )]));

    let popup_width = 64u16.min(area.width.saturating_sub(4));
    let content_height = lines.len() as u16 + 2;
    let popup_height = content_height.min(area.height.saturating_sub(4)).max(10);
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let max_scroll = (lines.len() as u16).saturating_sub(popup_height.saturating_sub(2));
    let scroll = app.help_overlay_scroll.min(max_scroll);

    let block = Block::default()
        .title(Span::styled(
            " Help ",
            Style::default().fg(NEON_YELLOW).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_YELLOW))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines).block(block).scroll((scroll, 0));

    frame.render_widget(paragraph, popup_area);
}

fn render_artifacts_list(app: &mut App, frame: &mut Frame) {
    let area = frame.area();

    let items: Vec<ListItem> = app
        .artifacts
        .iter()
        .enumerate()
        .map(|(i, artifact)| {
            let is_selected = i == app.selected_artifact;
            let icon = if artifact.content_type == "directory" {
                "📁"
            } else {
                "📄"
            };
            let size = artifact.display_size();
            let desc = if artifact.description.is_empty() {
                String::new()
            } else {
                format!("  — {}", artifact.description)
            };
            let run_ref = match (&artifact.run_project, &artifact.run_name) {
                (Some(proj), Some(rn)) => format!("  [run: {}/{}]", proj, rn),
                _ => String::new(),
            };
            let text = format!("{} {}  ({}){}{}", icon, artifact.name, size, desc, run_ref);

            let style = if is_selected {
                Style::default().fg(NEON_CYAN)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(text, style)))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_artifact));

    let mut title_spans = vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Runs", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("[Artifacts]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Infra", Style::default().fg(Color::DarkGray)),
        Span::styled(" ", Style::default()),
    ];
    if let Some(status) = &app.s3_pull_status {
        let color = if status.starts_with("Pull failed") || status.starts_with("S3 not") {
            NEON_MAGENTA
        } else if status.starts_with("Pulled") {
            NEON_GREEN
        } else {
            NEON_YELLOW
        };
        title_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        title_spans.push(Span::styled(status.clone(), Style::default().fg(color)));
    }

    let list = List::new(items)
        .block(
            Block::default()
                .title(Line::from(title_spans))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut state);

    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] details  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] download  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("?", Style::default().fg(NEON_YELLOW)),
        Span::styled("] help", Style::default().fg(Color::DarkGray)),
    ]);
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn render_artifact_detail(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let Some(artifact) = app.artifacts.get(app.selected_artifact) else {
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("Name:         ", Style::default().fg(DIM_CYAN)),
            Span::styled(&artifact.name, Style::default().fg(NEON_CYAN).bold()),
        ]),
        Line::from(vec![
            Span::styled("Description:  ", Style::default().fg(DIM_CYAN)),
            Span::styled(
                if artifact.description.is_empty() {
                    "—"
                } else {
                    &artifact.description
                },
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Type:         ", Style::default().fg(DIM_CYAN)),
            Span::styled(&artifact.content_type, Style::default().fg(NEON_GREEN)),
        ]),
        Line::from(vec![
            Span::styled("Size:         ", Style::default().fg(DIM_CYAN)),
            Span::styled(artifact.display_size(), Style::default().fg(NEON_YELLOW)),
        ]),
        Line::from(vec![
            Span::styled("Created:      ", Style::default().fg(DIM_CYAN)),
            Span::styled(
                artifact.created_at.as_deref().unwrap_or("—"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Updated:      ", Style::default().fg(DIM_CYAN)),
            Span::styled(
                artifact.updated_at.as_deref().unwrap_or("—"),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    if let (Some(proj), Some(rn)) = (&artifact.run_project, &artifact.run_name) {
        lines.push(Line::from(vec![
            Span::styled("Run:          ", Style::default().fg(DIM_CYAN)),
            Span::styled(
                format!("{}/{}", proj, rn),
                Style::default().fg(NEON_MAGENTA),
            ),
        ]));
    }

    if let Some(metadata) = &artifact.metadata
        && !metadata.is_empty()
    {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Metadata",
            Style::default().fg(NEON_CYAN).bold(),
        )));
        lines.push(Line::from(Span::styled(
            "─".repeat(60),
            Style::default().fg(DIM_CYAN),
        )));
        let obj = serde_json::Value::Object(metadata.clone());
        render_json_value_full(&obj, 1, &mut lines);
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("Files ({})", artifact.files.len()),
        Style::default().fg(NEON_CYAN).bold(),
    )));
    lines.push(Line::from(Span::styled(
        "─".repeat(60),
        Style::default().fg(DIM_CYAN),
    )));

    for file in &artifact.files {
        let size_str = match file.size_bytes {
            Some(b) if b >= 1_048_576 => format!("{:.1} MB", b as f64 / 1_048_576.0),
            Some(b) if b >= 1024 => format!("{:.1} KB", b as f64 / 1024.0),
            Some(b) => format!("{} B", b),
            None => "—".to_string(),
        };
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(&file.path, Style::default().fg(Color::White)),
            Span::styled(
                format!("  ({})", size_str),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let visible_height = chunks[0].height.saturating_sub(2) as usize;
    let max_scroll = lines.len().saturating_sub(visible_height);
    let scroll = app.artifact_detail_scroll.min(max_scroll);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
            Span::styled("Artifact: ", Style::default().fg(DIM_CYAN)),
            Span::styled(&artifact.name, Style::default().fg(NEON_CYAN).bold()),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((scroll as u16, 0));

    frame.render_widget(paragraph, chunks[0]);

    let mut footer_spans = vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Esc", Style::default().fg(NEON_YELLOW)),
        Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] download  ", Style::default().fg(Color::DarkGray)),
    ];
    if artifact.run_project.is_some() && artifact.run_name.is_some() {
        footer_spans.push(Span::styled("[", Style::default().fg(DIM_CYAN)));
        footer_spans.push(Span::styled("g", Style::default().fg(NEON_YELLOW)));
        footer_spans.push(Span::styled(
            "] go to run  ",
            Style::default().fg(Color::DarkGray),
        ));
    }
    footer_spans.push(Span::styled("[", Style::default().fg(DIM_CYAN)));
    footer_spans.push(Span::styled("?", Style::default().fg(NEON_YELLOW)));
    footer_spans.push(Span::styled("] help", Style::default().fg(Color::DarkGray)));
    frame.render_widget(Paragraph::new(Line::from(footer_spans)), chunks[1]);
}

fn render_infra_dashboard(app: &mut App, frame: &mut Frame) {
    let area = frame.area();

    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    render_infra_provider_tabs(app, frame, v_chunks[0]);

    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(v_chunks[1]);

    render_infra_instances_panel(app, frame, h_chunks[0]);
    render_infra_types_panel(app, frame, h_chunks[1]);
    render_infra_help_bar(app, frame, v_chunks[2]);

    if app.launch_selecting_region {
        render_infra_region_popup(app, frame);
    }
    if app.launch_confirming {
        render_infra_launch_confirm(app, frame);
    }
    if app.session_modal_open {
        render_session_modal(app, frame);
    }
    if app.add_machine_open {
        render_add_machine_modal(app, frame);
    }
}

fn render_infra_provider_tabs(app: &App, frame: &mut Frame, area: Rect) {
    let lambda_configured = app.infra_config.lambda_config.api_key.is_some();
    let vast_configured = app.infra_config.vast.api_key.is_some();
    let prime_configured = app.infra_config.prime.api_key.is_some();
    let local_configured = !app.infra_config.local.is_empty();

    let lambda_selected = app.selected_infra_provider == Provider::Lambda;
    let vast_selected = app.selected_infra_provider == Provider::Vast;
    let prime_selected = app.selected_infra_provider == Provider::Prime;
    let local_selected = app.selected_infra_provider == Provider::Local;

    let spans = vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Runs", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Artifacts", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("[Infra]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled("   Provider: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            if vast_selected { "[V Vast" } else { "V Vast" },
            if vast_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if vast_configured {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ),
        Span::styled(
            if vast_configured { " ✓" } else { " ✗" },
            Style::default().fg(if vast_configured {
                NEON_GREEN
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled(
            if vast_selected { "]" } else { "" },
            Style::default().fg(NEON_CYAN).bold(),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            if prime_selected {
                "[P Prime"
            } else {
                "P Prime"
            },
            if prime_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if prime_configured {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ),
        Span::styled(
            if prime_configured { " ✓" } else { " ✗" },
            Style::default().fg(if prime_configured {
                NEON_GREEN
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled(
            if prime_selected { "]" } else { "" },
            Style::default().fg(NEON_CYAN).bold(),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            if lambda_selected {
                "[λ Lambda"
            } else {
                "λ Lambda"
            },
            if lambda_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if lambda_configured {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ),
        Span::styled(
            if lambda_configured { " ✓" } else { " ✗" },
            Style::default().fg(if lambda_configured {
                NEON_GREEN
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled(
            if lambda_selected { "]" } else { "" },
            Style::default().fg(NEON_CYAN).bold(),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            if local_selected {
                "[L Local"
            } else {
                "L Local"
            },
            if local_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if local_configured {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ),
        Span::styled(
            if local_configured { " ✓" } else { " ✗" },
            Style::default().fg(if local_configured {
                NEON_GREEN
            } else {
                Color::DarkGray
            }),
        ),
        Span::styled(
            if local_selected { "]" } else { "" },
            Style::default().fg(NEON_CYAN).bold(),
        ),
        Span::styled("     ", Style::default()),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("c", Style::default().fg(NEON_CYAN)),
        Span::styled("] config", Style::default().fg(Color::DarkGray)),
    ];

    let line = Line::from(spans);
    let para = Paragraph::new(line);
    frame.render_widget(para, area);
}

fn render_infra_instances_panel(app: &App, frame: &mut Frame, area: Rect) {
    let instances = app.filtered_infra_instances();
    let is_focused = app.infra_active_panel == InfraPanel::Instances;

    let title = Line::from(vec![Span::styled(
        format!(" ◆ Active Instances ({}) ", instances.len()),
        Style::default()
            .fg(if is_focused { NEON_CYAN } else { Color::Gray })
            .bold(),
    )]);

    let border_color = if is_focused { NEON_CYAN } else { DIM_CYAN };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    if app.infra_loading {
        let loading = Paragraph::new(Span::styled("Loading...", Style::default().fg(NEON_YELLOW)))
            .block(block);
        frame.render_widget(loading, area);
        return;
    }

    if instances.is_empty() {
        let empty_text = if app.selected_infra_provider == Provider::Local {
            vec![
                Line::from(""),
                Line::from(Span::styled(
                    "No local machines configured",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Press [a] to add a machine",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "by SSH user and address.",
                    Style::default().fg(Color::DarkGray),
                )),
            ]
        } else {
            vec![
                Line::from(""),
                Line::from(Span::styled(
                    "No active instances",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Press → or Tab to browse",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "available instance types",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "and launch a new instance.",
                    Style::default().fg(Color::DarkGray),
                )),
            ]
        };
        let empty = Paragraph::new(empty_text).block(block);
        frame.render_widget(empty, area);
        return;
    }

    let mut instance_runs: HashMap<String, Vec<&Run>> = HashMap::new();
    for run in &app.runs {
        if run.is_running()
            && let Some(id) = run
                .config
                .as_ref()
                .and_then(|c| c.get("_instance_id"))
                .and_then(|v| v.as_str())
        {
            instance_runs.entry(id.to_string()).or_default().push(run);
        }
    }

    let items: Vec<ListItem> = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| {
            let is_selected = is_focused && i == app.selected_infra_instance;

            let (status_icon, status_color) = match instance.status {
                InstanceStatus::Running => ("●", NEON_GREEN),
                InstanceStatus::Booting | InstanceStatus::Pending => ("○", NEON_YELLOW),
                InstanceStatus::Stopping => ("◐", NEON_YELLOW),
                InstanceStatus::Stopped => ("○", Color::DarkGray),
                InstanceStatus::Terminated => ("×", Color::DarkGray),
                InstanceStatus::Error => ("!", NEON_MAGENTA),
            };

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if instance.status.is_active() {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::Gray)
            };

            let provider_char = match instance.provider {
                Provider::Lambda => "λ",
                Provider::Vast => "V",
                Provider::Prime => "P",
                Provider::Local => "L",
            };

            let ip_display = instance.ip.as_deref().unwrap_or("pending...");

            let line1 = Line::from(vec![
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(instance.display_name().to_string(), name_style),
                Span::styled("  ", Style::default()),
                Span::styled(provider_char, Style::default().fg(Color::DarkGray)),
                Span::styled("  ", Style::default()),
                Span::styled(&instance.instance_type, Style::default().fg(NEON_YELLOW)),
            ]);

            let mut line2_spans = vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    instance.status.as_str().to_string(),
                    Style::default().fg(status_color),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(&instance.region, Style::default().fg(Color::DarkGray)),
                Span::styled("  ", Style::default()),
                Span::styled(ip_display, Style::default().fg(NEON_CYAN)),
            ];
            if let Some(price) = instance.price_display() {
                line2_spans.push(Span::styled("  ", Style::default()));
                line2_spans.push(Span::styled(price, Style::default().fg(NEON_GREEN)));
            }
            let line2 = Line::from(line2_spans);

            let mut lines = vec![line1, line2];

            let composite_id = format!("{}:{}", instance.provider.as_str(), instance.id);
            if let Some(runs) = instance_runs.get(&composite_id) {
                let run_names: String = runs
                    .iter()
                    .map(|r| r.display_name())
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::from(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled("runs: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(run_names, Style::default().fg(NEON_MAGENTA)),
                ]));
            }

            ListItem::new(lines)
        })
        .collect();

    let mut state = ListState::default();
    if is_focused {
        state.select(Some(app.selected_infra_instance));
    }

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_infra_types_panel(app: &mut App, frame: &mut Frame, area: Rect) {
    let is_focused = app.infra_active_panel == InfraPanel::Types;

    let title = Line::from(vec![
        Span::styled(
            format!(" ◆ Instance Types ({}) ", app.infra_types.len()),
            Style::default()
                .fg(if is_focused { NEON_CYAN } else { Color::Gray })
                .bold(),
        ),
        Span::styled(
            format!("[sort: {}] ", app.infra_type_sort.label()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let border_color = if is_focused { NEON_CYAN } else { DIM_CYAN };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    if app.infra_loading {
        let loading = Paragraph::new(Span::styled("Loading...", Style::default().fg(NEON_YELLOW)))
            .block(block);
        frame.render_widget(loading, area);
        return;
    }

    if app.infra_types.is_empty() {
        let provider = app.selected_infra_provider;
        let msg = if provider == Provider::Local {
            "Local machines are managed in the instances panel.".to_string()
        } else {
            let config = app.infra_config.get_provider_config(provider);
            if config.api_key.is_none() {
                format!(
                    "No API key for {}. Press [c] to configure.",
                    provider.display_name()
                )
            } else {
                "No instance types available".to_string()
            }
        };
        let empty =
            Paragraph::new(Span::styled(msg, Style::default().fg(Color::DarkGray))).block(block);
        frame.render_widget(empty, area);
        return;
    }

    let items: Vec<ListItem> = app
        .infra_types
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let is_selected = is_focused && i == app.selected_infra_type;
            let has_availability = !it.regions.is_empty();

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else if has_availability {
                Style::default().fg(NEON_GREEN)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            let gpu_info = if let Some(gpu_desc) = &it.gpu_description {
                gpu_desc.clone()
            } else if let Some(gpu_name) = &it.gpu_name {
                format!("{}x {}", it.gpu_count, gpu_name)
            } else {
                format!("{}x GPU", it.gpu_count)
            };

            let region_info = if it.regions.is_empty() {
                "-".to_string()
            } else if it.regions.len() == 1 {
                it.regions[0].clone()
            } else {
                format!("{} regions", it.regions.len())
            };

            let spans = vec![
                Span::styled(format!("{:<22}", it.name), name_style),
                Span::styled(
                    format!("{:<30}", gpu_info),
                    if has_availability {
                        Style::default().fg(NEON_YELLOW)
                    } else {
                        Style::default().fg(Color::Rgb(80, 80, 40))
                    },
                ),
                Span::styled(
                    format!("{:<12}", it.price_display()),
                    if has_availability {
                        Style::default().fg(NEON_GREEN)
                    } else {
                        Style::default().fg(Color::Rgb(40, 80, 40))
                    },
                ),
                Span::styled(
                    region_info,
                    if has_availability {
                        Style::default().fg(Color::Gray)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ];

            ListItem::new(Line::from(spans))
        })
        .collect();

    if !is_focused {
        app.infra_types_list_state.select(None);
    }

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");
    frame.render_stateful_widget(list, area, &mut app.infra_types_list_state);
}

fn render_infra_help_bar(app: &App, frame: &mut Frame, area: Rect) {
    let is_local = app.selected_infra_provider == Provider::Local;
    let help = if app.add_machine_open
        || app.launch_selecting_region
        || app.launch_confirming
        || app.session_modal_open
    {
        Line::from(vec![])
    } else if app.infra_active_panel == InfraPanel::Instances && is_local {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] ssh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("a", Style::default().fg(NEON_GREEN)),
            Span::styled("] add  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("1-4", Style::default().fg(NEON_YELLOW)),
            Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("?", Style::default().fg(NEON_YELLOW)),
            Span::styled("] help", Style::default().fg(Color::DarkGray)),
        ])
    } else if app.infra_active_panel == InfraPanel::Instances {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Tab", Style::default().fg(NEON_CYAN)),
            Span::styled("] panel  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] ssh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("1-4", Style::default().fg(NEON_YELLOW)),
            Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("?", Style::default().fg(NEON_YELLOW)),
            Span::styled("] help", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Tab", Style::default().fg(NEON_CYAN)),
            Span::styled("] panel  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] launch  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("1-4", Style::default().fg(NEON_YELLOW)),
            Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("?", Style::default().fg(NEON_YELLOW)),
            Span::styled("] help", Style::default().fg(Color::DarkGray)),
        ])
    };

    if let Some(error) = &app.infra_error {
        let error_line = Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(error, Style::default().fg(NEON_YELLOW)),
        ]);
        frame.render_widget(Paragraph::new(error_line), area);
    } else {
        frame.render_widget(Paragraph::new(help), area);
    }
}

fn render_infra_region_popup(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let selected_type = app.infra_types.get(app.selected_infra_type);
    let regions: Vec<&str> = selected_type
        .map(|t| t.regions.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default();

    let type_name = selected_type.map(|t| t.name.as_str()).unwrap_or("unknown");

    let area = frame.area();
    let popup_width = 50u16.min(area.width.saturating_sub(4));
    let popup_height = (regions.len() as u16 + 5).min(area.height.saturating_sub(4));
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("Type: ", Style::default().fg(Color::DarkGray)),
            Span::styled(type_name, Style::default().fg(NEON_YELLOW)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Select Region:",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    for (i, region) in regions.iter().enumerate() {
        let is_selected = i == app.launch_selected_region;
        let prefix = if is_selected { "▶ " } else { "  " };
        let style = if is_selected {
            Style::default().fg(NEON_CYAN).bold()
        } else {
            Style::default().fg(Color::Gray)
        };

        let price_suffix = selected_type
            .and_then(|t| t.metadata.get(&format!("price:{}", region)))
            .and_then(|c| c.parse::<u32>().ok())
            .map(|cents| format!("  ${:.2}/hr", cents as f64 / 100.0))
            .unwrap_or_default();

        let mut spans = vec![Span::styled(format!("{}{}", prefix, region), style)];
        if !price_suffix.is_empty() {
            spans.push(Span::styled(price_suffix, Style::default().fg(NEON_GREEN)));
        }
        lines.push(Line::from(spans));
    }

    let block = Block::default()
        .title(Span::styled(
            " Select Region ",
            Style::default().fg(NEON_CYAN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(NEON_CYAN))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn render_infra_launch_confirm(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let selected_type = app.infra_types.get(app.selected_infra_type);
    let type_name = selected_type.map(|t| t.name.as_str()).unwrap_or("unknown");
    let region = selected_type
        .and_then(|t| t.regions.get(app.launch_selected_region))
        .map(|s| s.as_str())
        .unwrap_or("default");
    let type_price = selected_type
        .and_then(|t| {
            t.metadata
                .get(&format!("price:{}", region))
                .and_then(|c| c.parse::<u32>().ok())
                .map(|cents| format!("${:.2}/hr", cents as f64 / 100.0))
        })
        .unwrap_or_else(|| {
            selected_type
                .map(|t| t.price_display())
                .unwrap_or_else(|| "?".to_string())
        });

    let area = frame.area();
    let popup_width = 50u16.min(area.width.saturating_sub(4));
    let popup_height = 8u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Launch ", Style::default().fg(Color::White)),
            Span::styled(type_name, Style::default().fg(NEON_YELLOW).bold()),
            Span::styled(" in ", Style::default().fg(Color::White)),
            Span::styled(region, Style::default().fg(NEON_CYAN)),
            Span::styled("?", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Price: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&type_price, Style::default().fg(NEON_GREEN)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("y", Style::default().fg(NEON_GREEN)),
            Span::styled("] Yes  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("n", Style::default().fg(NEON_MAGENTA)),
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " Launch Instance ",
            Style::default().fg(NEON_GREEN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(NEON_GREEN))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, popup_area);
}

fn render_session_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 52u16.min(area.width.saturating_sub(4));
    let popup_height = 14u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let instance_name = app
        .session_modal_instance
        .as_ref()
        .map(|i| i.display_name().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let field_width = popup_width.saturating_sub(20) as usize;

    let truncate = |s: &str, max_len: usize| -> String {
        if s.len() > max_len {
            format!("...{}", &s[s.len().saturating_sub(max_len - 3)..])
        } else {
            s.to_string()
        }
    };

    let python_display = truncate(&app.session_python_version, field_width);
    let repo_display = truncate(&app.session_repo_path, field_width);
    let command_display = truncate(&app.session_command, field_width);

    let focused_style = Style::default().fg(NEON_CYAN).bold();
    let unfocused_style = Style::default().fg(Color::Gray);
    let label_style = Style::default().fg(Color::DarkGray);

    let py_style = if app.session_modal_focus == SessionModalField::PythonVersion {
        focused_style
    } else {
        unfocused_style
    };
    let repo_style = if app.session_modal_focus == SessionModalField::RepoPath {
        focused_style
    } else {
        unfocused_style
    };
    let cmd_style = if app.session_modal_focus == SessionModalField::Command {
        focused_style
    } else {
        unfocused_style
    };
    let tmux_style = if app.session_modal_focus == SessionModalField::SkipTmux {
        focused_style
    } else {
        unfocused_style
    };

    let checkbox = if app.session_skip_tmux { "[x]" } else { "[ ]" };

    let lines = vec![
        Line::from(vec![
            Span::styled("Instance: ", label_style),
            Span::styled(&instance_name, Style::default().fg(NEON_YELLOW)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Python version: ", label_style),
            Span::styled(format!("[{}]", python_display), py_style),
        ]),
        Line::from(vec![
            Span::styled("Local repo:     ", label_style),
            Span::styled(format!("[{}]", repo_display), repo_style),
        ]),
        Line::from(vec![
            Span::styled("Command:        ", label_style),
            Span::styled(format!("[{}]", command_display), cmd_style),
        ]),
        Line::from(Span::styled(
            "  (runs from extty-projects/<name> on remote)",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("{} ", checkbox), tmux_style),
            Span::styled("Skip tmux (already running)", tmux_style),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Tab", Style::default().fg(NEON_CYAN)),
            Span::styled("] next  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] Launch  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " SSH with Setup ",
            Style::default().fg(NEON_CYAN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(NEON_CYAN))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn render_add_machine_modal(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let area = frame.area();
    let popup_width = 50u16.min(area.width.saturating_sub(4));
    let popup_height = 12u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let fields: [(&str, &str, bool); 3] = [
        (
            "SSH User",
            &app.add_machine_user,
            app.add_machine_focus == AddMachineField::User,
        ),
        (
            "Host",
            &app.add_machine_host,
            app.add_machine_focus == AddMachineField::Host,
        ),
        (
            "Name",
            &app.add_machine_name,
            app.add_machine_focus == AddMachineField::Name,
        ),
    ];

    let mut lines: Vec<Line> = vec![Line::from("")];

    for (label, value, focused) in &fields {
        let arrow = if *focused { "→ " } else { "  " };
        let arrow_style = if *focused {
            Style::default().fg(NEON_MAGENTA)
        } else {
            Style::default()
        };
        let label_style = if *focused {
            Style::default().fg(NEON_CYAN).bold()
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let mut spans = vec![
            Span::styled(arrow, arrow_style),
            Span::styled(format!("{:<10}", label), label_style),
            Span::styled(*value, Style::default().fg(Color::White)),
        ];
        if *focused {
            spans.push(Span::styled("█", Style::default().fg(NEON_CYAN)));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  e.g. Host: 192.168.1.100 or myserver:2222",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  [", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] next  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_GREEN)),
        Span::styled("] save  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] cancel", Style::default().fg(Color::DarkGray)),
    ]));

    let block = Block::default()
        .title(Span::styled(
            " ◆ Add Local Machine ",
            Style::default().fg(NEON_CYAN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, popup_area);
}

fn render_infra_config(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let providers = Provider::all();

    let items: Vec<ListItem> = providers
        .iter()
        .enumerate()
        .map(|(i, provider)| {
            let is_selected = i == app.config_provider_index;

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else {
                Style::default().fg(Color::Gray)
            };

            if *provider == Provider::Local {
                let count = app.infra_config.local.len();
                let (status_icon, status_color) = if count > 0 {
                    ("✓", NEON_GREEN)
                } else {
                    ("✗", Color::DarkGray)
                };
                let detail = format!("{} machine(s)", count);
                let spans = vec![
                    Span::styled(
                        format!("{} ", status_icon),
                        Style::default().fg(status_color),
                    ),
                    Span::styled(format!("{:<12}", provider.display_name()), name_style),
                    Span::styled("  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        detail,
                        Style::default().fg(if count > 0 {
                            NEON_YELLOW
                        } else {
                            Color::DarkGray
                        }),
                    ),
                ];
                return ListItem::new(Line::from(spans));
            }

            let config = app.infra_config.get_provider_config(*provider);
            let has_key = config.api_key.is_some();

            let (status_icon, status_color) = if has_key {
                ("✓", NEON_GREEN)
            } else {
                ("✗", Color::DarkGray)
            };

            let key_display = if has_key {
                let key = config.api_key.as_ref().unwrap();
                if key.len() > 8 {
                    format!("{}...{}", &key[..4], &key[key.len() - 4..])
                } else {
                    "****".to_string()
                }
            } else {
                "not configured".to_string()
            };

            let mut spans = vec![
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(format!("{:<12}", provider.display_name()), name_style),
                Span::styled("  API Key: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    key_display,
                    Style::default().fg(if has_key {
                        NEON_YELLOW
                    } else {
                        Color::DarkGray
                    }),
                ),
            ];

            if is_selected && app.config_editing_key {
                spans.push(Span::styled("  → ", Style::default().fg(NEON_MAGENTA)));
                spans.push(Span::styled(
                    &app.config_api_key_input,
                    Style::default().fg(Color::White),
                ));
                spans.push(Span::styled("█", Style::default().fg(NEON_CYAN)));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.config_provider_index));

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled(
            "Provider Configuration",
            Style::default().fg(NEON_CYAN).bold(),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut state);

    let help = if app.config_editing_key {
        Line::from(vec![
            Span::styled("Type API key, then ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] save  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] cancel", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::styled("e", Style::default().fg(NEON_GREEN)),
            Span::styled("] edit key  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("?", Style::default().fg(NEON_YELLOW)),
            Span::styled("] help", Style::default().fg(Color::DarkGray)),
        ])
    };

    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn render_terminate_confirm(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let instance_name = app
        .pending_terminate_instance
        .and_then(|idx| {
            app.filtered_infra_instances()
                .get(idx)
                .map(|i| i.display_name().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let popup_height = 7u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Terminate instance ", Style::default().fg(Color::White)),
            Span::styled(&instance_name, Style::default().fg(NEON_CYAN).bold()),
            Span::styled("? (y/n)", Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "This will stop the instance and may delete data.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("y", Style::default().fg(NEON_GREEN)),
            Span::styled("] Yes  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("n", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] No  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " ⚠ CONFIRM TERMINATE ",
            Style::default().fg(NEON_MAGENTA).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_MAGENTA))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, popup_area);
}

fn render_complete_confirm(app: &App, frame: &mut Frame) {
    use ratatui::widgets::Clear;

    let run_name = app
        .runs
        .get(app.selected_run)
        .map(|r| r.name.as_str())
        .unwrap_or("unknown");

    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let popup_height = 6u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    frame.render_widget(Clear, popup_area);

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Mark ", Style::default().fg(Color::White)),
            Span::styled(run_name, Style::default().fg(NEON_CYAN).bold()),
            Span::styled(" as complete? (y/n)", Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("y", Style::default().fg(NEON_GREEN)),
            Span::styled("] Yes  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("n", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] No  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let block = Block::default()
        .title(Span::styled(
            " CONFIRM COMPLETE ",
            Style::default().fg(NEON_CYAN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(NEON_CYAN))
        .style(Style::default().bg(Color::Black));

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, popup_area);
}

fn render_s3_config(app: &App, frame: &mut Frame) {
    use ratatui::layout::Alignment;

    let area = frame.area();

    let fields: [(&str, String, bool); 6] = [
        ("bucket", app.s3_config.bucket.clone(), true),
        ("prefix", app.s3_config.prefix.clone(), false),
        (
            "region",
            app.s3_config.region.clone().unwrap_or_default(),
            false,
        ),
        (
            "access_key_id",
            app.s3_config.access_key_id.clone().unwrap_or_default(),
            false,
        ),
        (
            "secret_access_key",
            app.s3_config.secret_access_key.clone().unwrap_or_default(),
            false,
        ),
        (
            "endpoint_url",
            app.s3_config.endpoint_url.clone().unwrap_or_default(),
            false,
        ),
    ];

    let items: Vec<ListItem> = fields
        .iter()
        .enumerate()
        .map(|(i, (name, value, required))| {
            let is_selected = i == app.s3_config_field;
            let has_value = !value.is_empty();

            let (status_icon, status_color) = if has_value {
                ("✓", NEON_GREEN)
            } else if *required {
                ("✗", NEON_MAGENTA)
            } else {
                ("○", Color::DarkGray)
            };

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else {
                Style::default().fg(Color::Gray)
            };

            let display_value = if *name == "secret_access_key" && has_value {
                if value.len() > 8 {
                    format!("{}...{}", &value[..4], &value[value.len() - 4..])
                } else {
                    "****".to_string()
                }
            } else if has_value {
                value.clone()
            } else {
                "not set".to_string()
            };

            let mut spans = vec![
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(format!("{:<18}", name), name_style),
                Span::styled(
                    display_value,
                    Style::default().fg(if has_value {
                        NEON_YELLOW
                    } else {
                        Color::DarkGray
                    }),
                ),
            ];

            if is_selected && app.s3_config_editing {
                spans.clear();
                spans.push(Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ));
                spans.push(Span::styled(format!("{:<18}", name), name_style));
                spans.push(Span::styled(
                    &app.s3_config_input,
                    Style::default().fg(Color::White),
                ));
                spans.push(Span::styled("█", Style::default().fg(NEON_CYAN)));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.s3_config_field));

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("S3 Configuration", Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" (", Style::default().fg(Color::DarkGray)),
        Span::styled("~/.extty/s3/config.toml", Style::default().fg(Color::Gray)),
        Span::styled(")", Style::default().fg(Color::DarkGray)),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::Rgb(20, 20, 40)));

    frame.render_stateful_widget(list, area, &mut state);

    let help_text = if app.s3_config_editing {
        vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] Save  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Cancel", Style::default().fg(Color::DarkGray)),
        ]
    } else {
        vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("e/Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] Edit  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q/Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Back  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("?", Style::default().fg(NEON_YELLOW)),
            Span::styled("] help", Style::default().fg(Color::DarkGray)),
        ]
    };

    let mut help_line = help_text;
    if let Some(msg) = &app.s3_config_message {
        help_line.push(Span::styled("  ", Style::default()));
        help_line.push(Span::styled(msg, Style::default().fg(NEON_GREEN)));
    }

    let help = Paragraph::new(Line::from(help_line)).alignment(Alignment::Center);
    let help_area = ratatui::layout::Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };
    frame.render_widget(help, help_area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    #[test]
    fn confusion_matrix_card_renders_without_panic() {
        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let point = ConfusionMatrixPoint {
            step: 5,
            timestamp: 0.0,
            labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            matrix: vec![vec![10, 1, 0], vec![2, 8, 1], vec![0, 1, 9]],
        };
        terminal
            .draw(|f| {
                render_confusion_matrix_card(f, Rect::new(0, 0, 40, 12), "eval/cm", &point, true);
            })
            .unwrap();
    }

    #[test]
    #[ignore = "snapshot dump for manual visual inspection; run with --ignored"]
    fn focused_confusion_matrix_snapshot() {
        let backend = TestBackend::new(80, 22);
        let mut terminal = Terminal::new(backend).unwrap();
        let point = ConfusionMatrixPoint {
            step: 19,
            timestamp: 0.0,
            labels: vec![
                "cat".to_string(),
                "dog".to_string(),
                "fish".to_string(),
                "bird".to_string(),
                "frog".to_string(),
            ],
            matrix: vec![
                vec![43, 2, 3, 1, 2],
                vec![1, 41, 2, 0, 1],
                vec![0, 1, 44, 2, 1],
                vec![2, 0, 1, 40, 3],
                vec![1, 2, 0, 1, 42],
            ],
        };
        terminal
            .draw(|f| {
                render_focused_confusion_matrix(
                    f,
                    Rect::new(0, 0, 80, 22),
                    "eval/cm",
                    &point,
                    19,
                    20,
                );
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        println!("--- snapshot ---");
        for y in 0..buf.area.height {
            let mut row = String::new();
            for x in 0..buf.area.width {
                let symbol = buf[(x, y)].symbol();
                row.push_str(if symbol.is_empty() { " " } else { symbol });
            }
            println!("{}", row);
        }
        println!("--- end snapshot ---");
    }

    #[test]
    fn focused_confusion_matrix_renders_without_panic() {
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let point = ConfusionMatrixPoint {
            step: 5,
            timestamp: 0.0,
            labels: vec!["cat".to_string(), "dog".to_string(), "fish".to_string()],
            matrix: vec![vec![10, 1, 0], vec![2, 8, 1], vec![0, 1, 9]],
        };
        terminal
            .draw(|f| {
                render_focused_confusion_matrix(
                    f,
                    Rect::new(0, 0, 120, 30),
                    "eval/cm",
                    &point,
                    3,
                    20,
                );
            })
            .unwrap();
    }

    #[test]
    fn focused_confusion_matrix_tiny_area_does_not_panic() {
        let backend = TestBackend::new(20, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let point = ConfusionMatrixPoint {
            step: 5,
            timestamp: 0.0,
            labels: vec!["a".to_string(), "b".to_string()],
            matrix: vec![vec![0, 0], vec![0, 0]],
        };
        terminal
            .draw(|f| {
                render_focused_confusion_matrix(f, Rect::new(0, 0, 20, 8), "eval/cm", &point, 0, 1);
            })
            .unwrap();
    }
}
