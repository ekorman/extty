use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

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

mod data;
mod infra;
mod remote;
mod s3;
use data::{
    Evaluation, Example, MetricPoint, Model, Reward, Run, delete_evaluation, delete_model,
    load_all_evaluations, load_models, load_runs,
};
use infra::{
    InfraConfig, Instance, InstanceStatus, InstanceType, Provider, generate_script, get_provider,
    load_config, save_config,
};
use remote::RemoteSync;

enum SyncMessage {
    SyncCompleted,
}

enum SetupMessage {
    Status(String),
    Done(String),
    Error(String),
}

fn run_sync_loop(mut remote_sync: RemoteSync, tx: mpsc::Sender<SyncMessage>) {
    loop {
        thread::sleep(Duration::from_millis(500));

        if remote_sync.sync().is_ok() && tx.send(SyncMessage::SyncCompleted).is_err() {
            break;
        }
    }
}

// View mode: Runs, Models, or Infra
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Runs,
    Models,
    Infra,
}

// The views in our app
#[derive(Clone, Copy, PartialEq)]
enum View {
    List,
    RunDetail,
    ModelDetail,
    Focused,
    InfraList,
    InfraConfig,
    S3Config,
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

// A card in the detail grid - either a chart, example group, or evaluation
#[derive(Clone)]
enum Card {
    Chart { name: String },
    Examples { name: String },
    Evaluation { name: String },
}

// Represents an item in the hierarchical list view (runs)
#[derive(Clone, Debug)]
enum ListEntry {
    Project { name: String },
    Run { run_index: usize },
}

// Represents an item in the hierarchical list view (models)
#[derive(Clone, Debug)]
enum ModelListEntry {
    Project { name: String },
    Model { model_index: usize },
}

// All application state lives here
struct App {
    runs: Vec<Run>,
    models: Vec<Model>,
    model_evaluations: Vec<Evaluation>,
    selected_run: usize,
    selected_model: usize,
    selected_list_item: usize, // Current position in the flattened list (runs)
    selected_model_list_item: usize, // Current position in the flattened list (models)
    expanded_projects: HashSet<String>, // Which projects are expanded (runs)
    expanded_model_projects: HashSet<String>, // Which projects are expanded (models)
    selected_card: usize,
    selected_example: usize,  // Index within an example group when focused
    selected_prompt: usize,   // Index within prompts batch for an example
    selected_response: usize, // Index within response variants for an example
    scroll_offset: usize,
    focused_section: FocusedSection, // Which section (prompt/response) is focused
    prompt_scroll_offset: usize,     // Scroll offset for prompt in focused view
    response_scroll_offset: usize,   // Scroll offset for response in focused view
    view: View,
    view_mode: ViewMode,
    should_quit: bool,
    show_config: bool,
    show_delete_confirm: bool,
    pending_delete_run: Option<usize>, // Index into runs vector of run to delete
    pending_delete_model: Option<usize>, // Index into models vector of model to delete
    pending_delete_eval: Option<usize>, // Index into model_evaluations of eval to delete
    term_width: u16,
    term_height: u16,
    // Infra state
    infra_config: InfraConfig,
    infra_instances: Vec<Instance>,
    infra_types: Vec<InstanceType>,
    selected_infra_provider: Provider,
    selected_infra_instance: usize,
    selected_infra_type: usize,
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
    // Session setup modal
    session_modal_open: bool,
    session_modal_instance: Option<Instance>,
    session_python_version: String,
    session_repo_path: String,
    session_command: String,
    session_skip_tmux: bool,
    session_modal_focus: SessionModalField,
    setup_rx: Option<mpsc::Receiver<SetupMessage>>,
}

impl App {
    fn new() -> Self {
        let runs = load_runs();
        let models = load_models();
        let model_evaluations = load_all_evaluations();
        let infra_config = load_config().unwrap_or_default();
        let default_provider = infra_config.default_provider;
        App {
            runs,
            models,
            model_evaluations,
            selected_run: 0,
            selected_model: 0,
            selected_list_item: 0,
            selected_model_list_item: 0,
            expanded_projects: HashSet::new(),
            expanded_model_projects: HashSet::new(),
            selected_card: 0,
            selected_example: 0,
            selected_prompt: 0,
            selected_response: 0,
            scroll_offset: 0,
            focused_section: FocusedSection::Response,
            prompt_scroll_offset: 0,
            response_scroll_offset: 0,
            view: View::List,
            view_mode: ViewMode::Runs,
            should_quit: false,
            show_config: false,
            show_delete_confirm: false,
            pending_delete_run: None,
            pending_delete_model: None,
            pending_delete_eval: None,
            term_width: 80,
            term_height: 24,
            infra_config,
            infra_instances: Vec::new(),
            infra_types: Vec::new(),
            selected_infra_provider: default_provider,
            selected_infra_instance: 0,
            selected_infra_type: 0,
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
            session_modal_open: false,
            session_modal_instance: None,
            session_python_version: "3.12".to_string(),
            session_repo_path: String::new(),
            session_command: String::new(),
            session_skip_tmux: false,
            session_modal_focus: SessionModalField::default(),
            setup_rx: None,
        }
    }

    fn update_size(&mut self, width: u16, height: u16) {
        self.term_width = width;
        self.term_height = height;
    }

    // Build the flattened list of entries (projects and runs)
    fn list_entries(&self) -> Vec<ListEntry> {
        use std::collections::BTreeMap;

        // Group runs by project
        let mut projects: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for (i, run) in self.runs.iter().enumerate() {
            let project = run
                .project
                .clone()
                .unwrap_or_else(|| "(no project)".to_string());
            projects.entry(project).or_default().push(i);
        }

        let mut entries = Vec::new();

        for (project_name, run_indices) in projects {
            entries.push(ListEntry::Project {
                name: project_name.clone(),
            });

            if self.expanded_projects.contains(&project_name) {
                for run_index in run_indices {
                    entries.push(ListEntry::Run { run_index });
                }
            }
        }

        entries
    }

    // Build the flattened list of entries (projects and models)
    fn model_list_entries(&self) -> Vec<ModelListEntry> {
        use std::collections::BTreeMap;

        let mut projects: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for (i, model) in self.models.iter().enumerate() {
            projects.entry(model.project.clone()).or_default().push(i);
        }

        let mut entries = Vec::new();

        for (project_name, model_indices) in projects {
            entries.push(ModelListEntry::Project {
                name: project_name.clone(),
            });

            if self.expanded_model_projects.contains(&project_name) {
                for model_index in model_indices {
                    entries.push(ModelListEntry::Model { model_index });
                }
            }
        }

        entries
    }

    fn refresh_runs(&mut self) {
        let current_name = self.runs.get(self.selected_run).map(|r| r.name.clone());
        self.runs = load_runs();

        if let Some(name) = current_name {
            if let Some(idx) = self.runs.iter().position(|r| r.name == name) {
                self.selected_run = idx;
            } else {
                self.selected_run = self.selected_run.min(self.runs.len().saturating_sub(1));
            }
        }

        let entries = self.list_entries();
        self.selected_list_item = self.selected_list_item.min(entries.len().saturating_sub(1));
    }

    fn refresh_models(&mut self) {
        let current_name = self.models.get(self.selected_model).map(|m| m.name.clone());
        self.models = load_models();
        self.model_evaluations = load_all_evaluations();

        if let Some(name) = current_name {
            if let Some(idx) = self.models.iter().position(|m| m.name == name) {
                self.selected_model = idx;
            } else {
                self.selected_model = self.selected_model.min(self.models.len().saturating_sub(1));
            }
        }

        let entries = self.model_list_entries();
        self.selected_model_list_item = self
            .selected_model_list_item
            .min(entries.len().saturating_sub(1));
    }

    fn refresh_current_run(&mut self) {
        if let Some(run) = self.runs.get(self.selected_run) {
            let path = run.path.clone();
            if let Some(updated) = data::reload_run(&path) {
                self.runs[self.selected_run] = updated;
            }
        }
    }

    fn refresh_infra(&mut self) {
        self.infra_loading = true;
        self.infra_error = None;
        self.infra_instances.clear();

        let provider = self.selected_infra_provider;
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

            let mut rsync_cmd = std::process::Command::new("rsync");
            rsync_cmd
                .args([
                    "-az",
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
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());

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

            rsync_cmd
                .arg(&repo_with_slash)
                .arg(format!("{}@{}:~/project/", ssh_user, host));

            match rsync_cmd.status() {
                Ok(status) if status.success() => {
                    let _ = tx.send(SetupMessage::Status(
                        "Code synced, copying config...".to_string(),
                    ));
                }
                Ok(status) => {
                    let _ = tx.send(SetupMessage::Error(format!(
                        "rsync failed: exit {}",
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

            let _ = tx.send(SetupMessage::Status("Uploading script...".to_string()));

            let script = generate_script(&python_version, &command, skip_tmux);

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

        let config_width = 35u16;
        let has_config = self
            .current_run()
            .map(|r| r.config.is_some())
            .unwrap_or(false);
        let effective_width = if self.show_config && has_config {
            self.term_width.saturating_sub(config_width)
        } else {
            self.term_width
        };

        let cols = (effective_width / card_width).max(1) as usize;
        let visible_rows = (grid_height / card_height).max(1) as usize;
        (visible_rows, cols)
    }

    fn current_run(&self) -> Option<&Run> {
        self.runs.get(self.selected_run)
    }

    fn current_model(&self) -> Option<&Model> {
        self.models.get(self.selected_model)
    }

    fn cards(&self) -> Vec<Card> {
        let Some(run) = self.current_run() else {
            return vec![];
        };

        let mut cards = Vec::new();

        let mut metric_names: Vec<&String> = run.metrics.keys().collect();
        metric_names.sort();
        for name in metric_names {
            cards.push(Card::Chart { name: name.clone() });
        }

        let mut example_names: Vec<&String> = run.examples.keys().collect();
        example_names.sort();
        for name in example_names {
            cards.push(Card::Examples { name: name.clone() });
        }

        cards
    }

    fn model_cards(&self) -> Vec<Card> {
        let Some(model) = self.current_model() else {
            return vec![];
        };

        let mut cards = Vec::new();

        let model_evaluations: Vec<&Evaluation> = self
            .model_evaluations
            .iter()
            .filter(|e| e.model_name == model.name && e.project == model.project)
            .collect();

        for eval in model_evaluations {
            cards.push(Card::Evaluation {
                name: eval.name.clone(),
            });
        }

        cards
    }

    fn get_model_evaluation(&self, name: &str) -> Option<&Evaluation> {
        let model = self.current_model()?;
        self.model_evaluations
            .iter()
            .find(|e| e.model_name == model.name && e.project == model.project && e.name == name)
    }

    fn card_count(&self) -> usize {
        let Some(run) = self.current_run() else {
            return 0;
        };
        run.metrics.len() + run.examples.len()
    }

    fn model_card_count(&self) -> usize {
        let Some(model) = self.current_model() else {
            return 0;
        };
        self.model_evaluations
            .iter()
            .filter(|e| e.model_name == model.name && e.project == model.project)
            .count()
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
            View::ModelDetail => self.model_cards(),
            View::Focused => match self.view_mode {
                ViewMode::Runs => self.cards(),
                ViewMode::Models => self.model_cards(),
                ViewMode::Infra => vec![],
            },
            View::InfraList | View::InfraConfig | View::S3Config => vec![],
        }
    }

    // Get max scroll offset for prompt/response in focused view
    fn focused_max_scroll(&self, is_prompt: bool) -> usize {
        let cards = self.active_cards();
        let current_card = cards.get(self.selected_card);

        let total_height = self.term_height.saturating_sub(6) as usize;
        let visible_height = if is_prompt {
            (total_height * 40 / 100).saturating_sub(2)
        } else {
            (total_height * 60 / 100).saturating_sub(2)
        };

        let wrap_width = self.term_width.saturating_sub(4) as usize;

        let text = match current_card {
            Some(Card::Examples { name }) => {
                if let Some(examples) = self.current_run().and_then(|r| r.examples.get(name)) {
                    if let Some(example) = examples.get(self.selected_example) {
                        if is_prompt {
                            example
                                .prompts
                                .get(self.selected_prompt)
                                .cloned()
                                .unwrap_or_default()
                        } else {
                            example
                                .responses
                                .get(self.selected_prompt)
                                .and_then(|r| r.get(self.selected_response))
                                .cloned()
                                .unwrap_or_default()
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            }
            Some(Card::Evaluation { name }) => {
                let eval = match self.view_mode {
                    ViewMode::Runs | ViewMode::Infra => None,
                    ViewMode::Models => self.get_model_evaluation(name),
                };
                if let Some(eval) = eval {
                    if let Some(ex) = eval.examples.get(self.selected_example) {
                        if is_prompt {
                            ex.prompts
                                .get(self.selected_prompt)
                                .cloned()
                                .unwrap_or_default()
                        } else {
                            ex.responses
                                .get(self.selected_prompt)
                                .and_then(|r| r.get(self.selected_response))
                                .cloned()
                                .unwrap_or_default()
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

        match self.view {
            View::List => match self.view_mode {
                ViewMode::Runs => self.handle_list_key(code),
                ViewMode::Models => self.handle_model_list_key(code),
                ViewMode::Infra => self.handle_infra_list_key(code, modifiers),
            },
            View::RunDetail => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_detail_key(code, visible_rows, cols);
            }
            View::ModelDetail => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_model_detail_key(code, visible_rows, cols);
            }
            View::Focused => self.handle_focused_key(code),
            View::InfraList => self.handle_infra_list_key(code, modifiers),
            View::InfraConfig => self.handle_infra_config_key(code),
            View::S3Config => self.handle_s3_config_key(code),
        }
    }

    fn handle_delete_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some(run_idx) = self.pending_delete_run
                    && let Some(run) = self.runs.get(run_idx)
                {
                    let path = run.path.clone();
                    let _ = data::delete_run(&path);
                    self.refresh_runs();
                    if self.view == View::RunDetail {
                        self.view = View::List;
                    }
                }
                if let Some(model_idx) = self.pending_delete_model
                    && let Some(model) = self.models.get(model_idx)
                {
                    let path = model.path.clone();
                    let _ = delete_model(&path);
                    self.refresh_models();
                    if self.view == View::ModelDetail {
                        self.view = View::List;
                    }
                }
                if let Some(eval_idx) = self.pending_delete_eval
                    && let Some(eval) = self.model_evaluations.get(eval_idx)
                {
                    let path = eval.path.clone();
                    let _ = delete_evaluation(&path);
                    self.refresh_models();
                    self.selected_card = self.selected_card.saturating_sub(1);
                }
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
                self.pending_delete_model = None;
                self.pending_delete_eval = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
                self.pending_delete_model = None;
                self.pending_delete_eval = None;
            }
            _ => {}
        }
    }

    fn handle_list_key(&mut self, code: KeyCode) {
        let entries = self.list_entries();
        let entry_count = entries.len();

        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('m') => {
                self.view_mode = ViewMode::Models;
                self.selected_card = 0;
                self.scroll_offset = 0;
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
            }
            KeyCode::Down if self.selected_list_item < entry_count.saturating_sub(1) => {
                self.selected_list_item += 1;
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
                }
                None => {}
            },
            KeyCode::Char('d') => {
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    self.pending_delete_run = Some(*run_index);
                    self.show_delete_confirm = true;
                }
            }
            _ => {}
        }
    }

    fn handle_model_list_key(&mut self, code: KeyCode) {
        let entries = self.model_list_entries();
        let entry_count = entries.len();

        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('r') => {
                self.view_mode = ViewMode::Runs;
                self.selected_card = 0;
                self.scroll_offset = 0;
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
            KeyCode::Up if self.selected_model_list_item > 0 => {
                self.selected_model_list_item -= 1;
            }
            KeyCode::Down if self.selected_model_list_item < entry_count.saturating_sub(1) => {
                self.selected_model_list_item += 1;
            }
            KeyCode::Tab => {
                if let Some(ModelListEntry::Project { name }) =
                    entries.get(self.selected_model_list_item)
                {
                    if self.expanded_model_projects.contains(name) {
                        self.expanded_model_projects.remove(name);
                    } else {
                        self.expanded_model_projects.insert(name.clone());
                    }
                }
            }
            KeyCode::Enter => match entries.get(self.selected_model_list_item) {
                Some(ModelListEntry::Project { name }) => {
                    if self.expanded_model_projects.contains(name) {
                        self.expanded_model_projects.remove(name);
                    } else {
                        self.expanded_model_projects.insert(name.clone());
                    }
                }
                Some(ModelListEntry::Model { model_index }) => {
                    self.selected_model = *model_index;
                    self.selected_card = 0;
                    self.view = View::ModelDetail;
                }
                None => {}
            },
            KeyCode::Char('d') => {
                if let Some(ModelListEntry::Model { model_index }) =
                    entries.get(self.selected_model_list_item)
                {
                    self.pending_delete_model = Some(*model_index);
                    self.show_delete_confirm = true;
                }
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
                self.view = View::Focused;
            }
            KeyCode::Char('[') if self.selected_run > 0 => {
                self.selected_run -= 1;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char(']') if self.selected_run < self.runs.len().saturating_sub(1) => {
                self.selected_run += 1;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char('c') => {
                self.show_config = !self.show_config;
            }
            KeyCode::Char('d') if !self.runs.is_empty() => {
                self.pending_delete_run = Some(self.selected_run);
                self.show_delete_confirm = true;
            }
            _ => {}
        }
    }

    fn handle_model_detail_key(&mut self, code: KeyCode, visible_rows: usize, cols: usize) {
        let card_count = self.model_card_count();
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
                self.view = View::Focused;
            }
            KeyCode::Char('[') if self.selected_model > 0 => {
                self.selected_model -= 1;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char(']') if self.selected_model < self.models.len().saturating_sub(1) => {
                self.selected_model += 1;
                self.selected_card = 0;
                self.scroll_offset = 0;
            }
            KeyCode::Char('c') => {
                self.show_config = !self.show_config;
            }
            KeyCode::Char('d') if card_count > 0 => {
                let cards = self.model_cards();
                if let Some(Card::Evaluation { name }) = cards.get(self.selected_card)
                    && let Some(model) = self.current_model()
                    && let Some(eval_idx) = self.model_evaluations.iter().position(|e| {
                        e.name == *name && e.model_name == model.name && e.project == model.project
                    })
                {
                    self.pending_delete_eval = Some(eval_idx);
                    self.show_delete_confirm = true;
                }
            }
            _ => {}
        }
    }

    fn handle_focused_key(&mut self, code: KeyCode) {
        let (card_count, cards) = match self.view_mode {
            ViewMode::Runs => (self.card_count(), self.cards()),
            ViewMode::Models => (self.model_card_count(), self.model_cards()),
            ViewMode::Infra => (0, vec![]),
        };
        let current_card = cards.get(self.selected_card);

        let example_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .map(|e| e.len())
                .unwrap_or(0),
            Some(Card::Evaluation { name }) => match self.view_mode {
                ViewMode::Runs | ViewMode::Infra => 0,
                ViewMode::Models => self
                    .get_model_evaluation(name)
                    .map(|e| e.examples.len())
                    .unwrap_or(0),
            },
            _ => 0,
        };

        let prompt_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .and_then(|e| e.get(self.selected_example))
                .map(|ex| ex.prompts.len())
                .unwrap_or(0),
            Some(Card::Evaluation { name }) => self
                .get_model_evaluation(name)
                .and_then(|e| e.examples.get(self.selected_example))
                .map(|ex| ex.prompts.len())
                .unwrap_or(0),
            _ => 0,
        };

        let response_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .and_then(|e| e.get(self.selected_example))
                .and_then(|ex| ex.responses.get(self.selected_prompt))
                .map(|r| r.len())
                .unwrap_or(0),
            Some(Card::Evaluation { name }) => self
                .get_model_evaluation(name)
                .and_then(|e| e.examples.get(self.selected_example))
                .and_then(|ex| ex.responses.get(self.selected_prompt))
                .map(|r| r.len())
                .unwrap_or(0),
            _ => 0,
        };

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.view = match self.view_mode {
                    ViewMode::Runs => View::RunDetail,
                    ViewMode::Models => View::ModelDetail,
                    ViewMode::Infra => View::InfraList,
                };
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            // Tab toggles focus between prompt and response sections
            KeyCode::Tab => {
                self.focused_section = match self.focused_section {
                    FocusedSection::Prompt => FocusedSection::Response,
                    FocusedSection::Response => FocusedSection::Prompt,
                };
            }
            // k scrolls down, j scrolls up in the focused section
            KeyCode::Char('k') => match self.focused_section {
                FocusedSection::Prompt => {
                    let max = self.focused_max_scroll(true);
                    self.prompt_scroll_offset = (self.prompt_scroll_offset + 1).min(max);
                }
                FocusedSection::Response => {
                    let max = self.focused_max_scroll(false);
                    self.response_scroll_offset = (self.response_scroll_offset + 1).min(max);
                }
            },
            KeyCode::Char('j') => match self.focused_section {
                FocusedSection::Prompt => {
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_sub(1);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_sub(1);
                }
            },
            // PageDown/PageUp for faster scrolling
            KeyCode::PageDown => match self.focused_section {
                FocusedSection::Prompt => {
                    let max = self.focused_max_scroll(true);
                    self.prompt_scroll_offset = (self.prompt_scroll_offset + 10).min(max);
                }
                FocusedSection::Response => {
                    let max = self.focused_max_scroll(false);
                    self.response_scroll_offset = (self.response_scroll_offset + 10).min(max);
                }
            },
            KeyCode::PageUp => match self.focused_section {
                FocusedSection::Prompt => {
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_sub(10);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_sub(10);
                }
            },
            // Left/Right navigate between cards
            KeyCode::Left if card_count > 1 && self.selected_card > 0 => {
                self.selected_card -= 1;
                self.selected_example = 0;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            KeyCode::Right
                if card_count > 1 && self.selected_card < card_count.saturating_sub(1) =>
            {
                self.selected_card += 1;
                self.selected_example = 0;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            // Up/Down navigate within example groups
            KeyCode::Up if example_count > 0 && self.selected_example > 0 => {
                self.selected_example -= 1;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            KeyCode::Down
                if example_count > 0 && self.selected_example < example_count.saturating_sub(1) =>
            {
                self.selected_example += 1;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            // [ and ] navigate between prompts in batch
            KeyCode::Char('[') if prompt_count > 1 && self.selected_prompt > 0 => {
                self.selected_prompt -= 1;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
            }
            KeyCode::Char(']')
                if prompt_count > 1 && self.selected_prompt < prompt_count.saturating_sub(1) =>
            {
                self.selected_prompt += 1;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
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
            // c toggles model config panel (Models view only)
            KeyCode::Char('c') if self.view_mode == ViewMode::Models => {
                self.show_config = !self.show_config;
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

    fn handle_infra_list_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
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
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('r') => {
                self.view_mode = ViewMode::Runs;
                self.view = View::List;
            }
            KeyCode::Char('m') => {
                self.view_mode = ViewMode::Models;
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
                }
                _ => {}
            },
            KeyCode::Char('1') => {
                self.selected_infra_provider = Provider::Lambda;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('2') => {
                self.selected_infra_provider = Provider::Vast;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
                self.refresh_infra();
                self.refresh_infra_types();
            }
            KeyCode::Char('3') => {
                self.selected_infra_provider = Provider::Prime;
                self.selected_infra_instance = 0;
                self.selected_infra_type = 0;
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
                    self.pending_terminate_instance = Some(self.selected_infra_instance);
                    self.show_terminate_confirm = true;
                }
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
                    self.config_editing_key = true;
                    let provider = providers[self.config_provider_index];
                    let current_key = match provider {
                        Provider::Lambda => &self.infra_config.lambda_config.api_key,
                        Provider::Vast => &self.infra_config.vast.api_key,
                        Provider::Prime => &self.infra_config.prime.api_key,
                    };
                    self.config_api_key_input = current_key.clone().unwrap_or_default();
                }
                KeyCode::Char('d') => {
                    let provider = providers[self.config_provider_index];
                    match provider {
                        Provider::Lambda => self.infra_config.lambda_config.api_key = None,
                        Provider::Vast => self.infra_config.vast.api_key = None,
                        Provider::Prime => self.infra_config.prime.api_key = None,
                    }
                    let _ = save_config(&self.infra_config);
                }
                _ => {}
            }
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
        };

        let client = get_provider(provider, api_key);
        match client.launch(&opts) {
            Ok(ids) => {
                self.infra_error = None;
                self.refresh_infra();
                self.infra_active_panel = InfraPanel::Instances;
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
    }
}

fn run_tui(options: TuiOptions) -> Result<()> {
    let sync_rx: Option<Receiver<SyncMessage>> = if let Some(remote_url) = options.remote_url {
        let runs_dir = remote_runs_dir();
        let token = options
            .token
            .or_else(|| std::env::var("EX_REMOTE_TOKEN").ok());
        let mut remote_sync = RemoteSync::new(remote_url, token, runs_dir)?;
        remote_sync.sync()?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || run_sync_loop(remote_sync, tx));
        Some(rx)
    } else {
        None
    };

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

        // Check for sync messages from background thread (non-blocking)
        if let Some(rx) = &sync_rx
            && let Ok(SyncMessage::SyncCompleted) = rx.try_recv()
        {
            if matches!(
                app.view,
                View::RunDetail | View::ModelDetail | View::Focused
            ) {
                app.refresh_current_run();
            } else {
                app.refresh_runs();
                app.refresh_models();
            }
        }

        // Periodic refresh of run and model lists (less frequent)
        if last_list_refresh.elapsed() >= list_refresh_interval {
            app.refresh_runs();
            app.refresh_models();
            last_list_refresh = Instant::now();
        }

        // Auto-refresh infra instances when viewing infra tab (every 5 seconds)
        if app.view_mode == ViewMode::Infra
            && !app.show_config
            && app.infra_last_refresh.elapsed() >= Duration::from_secs(5)
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

        // Draw the UI
        terminal.draw(|frame| render(&app, frame))?;

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
}

struct TuiOptions {
    remote_url: Option<String>,
    token: Option<String>,
}

struct SyncOptions {
    target: Option<String>,
    force: bool,
    dry_run: bool,
}

fn parse_command() -> Result<Command> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Ok(Command::Tui(TuiOptions {
            remote_url: None,
            token: None,
        }));
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
        "--remote" | "--token" => {
            let opts = parse_tui_options(&mut args)?;
            Ok(Command::Tui(opts))
        }
        other if other.starts_with('-') => Err(anyhow::anyhow!("Unknown option: {}", other)),
        _ => Err(anyhow::anyhow!(
            "Unknown command: {}. Valid commands: pull, push, sync",
            args[0]
        )),
    }
}

fn parse_tui_options(args: &mut Vec<String>) -> Result<TuiOptions> {
    let mut remote_url = None;
    let mut token = None;

    while !args.is_empty() {
        match args[0].as_str() {
            "--remote" => {
                args.remove(0);
                remote_url = Some(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--remote requires a URL"))?
                        .clone(),
                );
                args.remove(0);
            }
            "--token" => {
                args.remove(0);
                token = Some(
                    args.first()
                        .ok_or_else(|| anyhow::anyhow!("--token requires a value"))?
                        .clone(),
                );
                args.remove(0);
            }
            other => {
                return Err(anyhow::anyhow!("Unknown option: {}", other));
            }
        }
    }

    Ok(TuiOptions { remote_url, token })
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

fn remote_runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".extty")
        .join("remote_runs")
}

fn render(app: &App, frame: &mut Frame) {
    match app.view {
        View::List => match app.view_mode {
            ViewMode::Runs => render_runs_list(app, frame),
            ViewMode::Models => render_models_list(app, frame),
            ViewMode::Infra => render_infra_dashboard(app, frame),
        },
        View::RunDetail => render_run_detail(app, frame),
        View::ModelDetail => render_model_detail(app, frame),
        View::Focused => render_focused(app, frame),
        View::InfraList => render_infra_dashboard(app, frame),
        View::InfraConfig => render_infra_config(app, frame),
        View::S3Config => render_s3_config(app, frame),
    }

    if app.show_delete_confirm {
        render_delete_confirm(app, frame);
    }
    if app.show_terminate_confirm {
        render_terminate_confirm(app, frame);
    }
}

fn render_runs_list(app: &App, frame: &mut Frame) {
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

                    // Count runs in this project
                    let run_count = app
                        .runs
                        .iter()
                        .filter(|r| r.project.as_deref().unwrap_or("(no project)") == name)
                        .count();

                    // Check if any runs in this project are running
                    let has_running = app
                        .runs
                        .iter()
                        .filter(|r| r.project.as_deref().unwrap_or("(no project)") == name)
                        .any(|r| r.is_running());

                    let name_style = if is_selected {
                        Style::default().fg(NEON_MAGENTA).bold()
                    } else if has_running {
                        Style::default().fg(NEON_GREEN).bold()
                    } else {
                        Style::default().fg(NEON_CYAN).bold()
                    };

                    ListItem::new(Line::from(vec![
                        Span::styled(icon, Style::default().fg(NEON_MAGENTA)),
                        Span::styled(name.clone(), name_style),
                        Span::styled(
                            format!("  ({} runs)", run_count),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]))
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
                            .map(|t| t.format("%H:%M").to_string())
                            .unwrap_or_else(|| "—".to_string())
                    };

                    let time_style = if is_running {
                        Style::default().fg(NEON_YELLOW)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };

                    // Indent runs under their project with tree branch
                    let mut spans = vec![
                        Span::styled("  └─ ", Style::default().fg(DIM_CYAN)),
                        Span::styled(status_icon, Style::default().fg(status_color)),
                        Span::styled(run.name.clone(), name_style),
                        Span::styled("  ", Style::default()),
                        Span::styled(start_str, Style::default().fg(Color::DarkGray)),
                        Span::styled(" → ", Style::default().fg(DIM_CYAN)),
                        Span::styled(end_str, time_style),
                    ];
                    if let Some(url) = &run.remote_url {
                        spans.push(Span::styled("  @ ", Style::default().fg(DIM_CYAN)));
                        spans.push(Span::styled(
                            url.clone(),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    ListItem::new(Line::from(spans))
                }
            }
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_list_item));

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("[Runs]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Models", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Infra", Style::default().fg(Color::DarkGray)),
        Span::styled(" ", Style::default()),
    ]);

    let list = List::new(items)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut state);

    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("m", Style::default().fg(NEON_YELLOW)),
        Span::styled("] models  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("i", Style::default().fg(NEON_YELLOW)),
        Span::styled("] infra  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
        Span::styled("] nav  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] expand  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] delete  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] quit", Style::default().fg(Color::DarkGray)),
    ]);
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn render_models_list(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let entries = app.model_list_entries();

    let items: Vec<ListItem> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let is_selected = i == app.selected_model_list_item;

            match entry {
                ModelListEntry::Project { name } => {
                    let is_expanded = app.expanded_model_projects.contains(name);
                    let icon = if is_expanded { "▼ " } else { "▶ " };

                    let model_count = app.models.iter().filter(|m| &m.project == name).count();

                    let name_style = if is_selected {
                        Style::default().fg(NEON_MAGENTA).bold()
                    } else {
                        Style::default().fg(NEON_CYAN).bold()
                    };

                    ListItem::new(Line::from(vec![
                        Span::styled(icon, Style::default().fg(NEON_MAGENTA)),
                        Span::styled(name.clone(), name_style),
                        Span::styled(
                            format!("  ({} models)", model_count),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]))
                }
                ModelListEntry::Model { model_index } => {
                    let model = &app.models[*model_index];

                    let model_evals: Vec<_> = app
                        .model_evaluations
                        .iter()
                        .filter(|e| e.model_name == model.name && e.project == model.project)
                        .collect();
                    let eval_count = model_evals.len();

                    // Find the latest evaluation timestamp (prefer started_at, fall back to logged_at)
                    let latest_eval_time = model_evals
                        .iter()
                        .filter_map(|e| e.started_at.or(e.logged_at))
                        .max();

                    let name_style = if is_selected {
                        Style::default().fg(NEON_CYAN).bold()
                    } else {
                        Style::default().fg(Color::Gray)
                    };

                    let mut spans = vec![
                        Span::styled("  └─ ", Style::default().fg(DIM_CYAN)),
                        Span::styled("◆ ", Style::default().fg(NEON_YELLOW)),
                        Span::styled(model.name.clone(), name_style),
                        Span::styled(
                            format!("  ({} evals)", eval_count),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ];

                    if let Some(latest) = latest_eval_time {
                        spans.push(Span::styled("  ", Style::default()));
                        spans.push(Span::styled(
                            latest.format("%m-%d %H:%M").to_string(),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }

                    ListItem::new(Line::from(spans))
                }
            }
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_model_list_item));

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Runs", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("[Models]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Infra", Style::default().fg(Color::DarkGray)),
        Span::styled(" ", Style::default()),
    ]);

    let list = List::new(items)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut state);

    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("r", Style::default().fg(NEON_YELLOW)),
        Span::styled("] runs  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("i", Style::default().fg(NEON_YELLOW)),
        Span::styled("] infra  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
        Span::styled("] nav  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] expand  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] delete  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] quit", Style::default().fg(Color::DarkGray)),
    ]);
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn render_run_detail(app: &App, frame: &mut Frame) {
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

    let total_examples: usize = run.examples.values().map(|v| v.len()).sum();
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
    if let Some(url) = &run.remote_url {
        header_spans.push(Span::styled("  │  ", Style::default().fg(DIM_CYAN)));
        header_spans.push(Span::styled(
            url.clone(),
            Style::default().fg(Color::DarkGray),
        ));
    }
    header_spans.push(Span::styled(
        &scroll_indicator,
        Style::default().fg(NEON_MAGENTA),
    ));
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
    if let Some(config_area) = config_area
        && let Some(config) = &run.config
    {
        render_config_panel(frame, config_area, config);
    }

    // Footer with styled keys
    let config_hint = if app.show_config { "hide" } else { "config" };
    let footer = Line::from(vec![
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
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] delete  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("c", Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("] {}  ", config_hint),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("[]", Style::default().fg(NEON_YELLOW)),
        Span::styled("] run ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}/{}", app.selected_run + 1, app.runs.len()),
            Style::default().fg(NEON_YELLOW),
        ),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[2]);
}

fn render_model_detail(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let Some(model) = app.current_model() else {
        frame.render_widget(Paragraph::new("No model selected"), area);
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

    let config_width = 35u16;
    let (grid_area, config_area) = if app.show_config && model.config.is_some() {
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(config_width)])
            .split(chunks[1]);
        (h_chunks[0], Some(h_chunks[1]))
    } else {
        (chunks[1], None)
    };

    let card_width = 40u16;
    let card_height = 12u16;
    let cols = (grid_area.width / card_width).max(1) as usize;
    let cards = app.model_cards();
    let total_cards = cards.len();
    let total_rows = total_cards.div_ceil(cols);
    let visible_rows = (grid_area.height / card_height) as usize;
    let max_scroll = total_rows.saturating_sub(visible_rows);
    let scroll = app.scroll_offset.min(max_scroll);

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

    let header_spans = vec![
        Span::styled("◆ ", Style::default().fg(NEON_YELLOW)),
        Span::styled(
            format!("{}/{}", model.project, model.name),
            Style::default().fg(NEON_CYAN).bold(),
        ),
        Span::styled("  │  ", Style::default().fg(DIM_CYAN)),
        Span::styled(format!("{}", total_cards), Style::default().fg(NEON_GREEN)),
        Span::styled(" evaluations", Style::default().fg(Color::DarkGray)),
        Span::styled(&scroll_indicator, Style::default().fg(NEON_MAGENTA)),
    ];
    let header_text = Line::from(header_spans);
    let header = Paragraph::new(header_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN)),
    );
    frame.render_widget(header, chunks[0]);

    render_model_cards_grid(app, frame, grid_area, &cards);

    if let Some(config_area) = config_area
        && let Some(config) = &model.config
    {
        render_config_panel(frame, config_area, config);
    }

    let config_hint = if app.show_config { "hide" } else { "config" };
    let footer = Line::from(vec![
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
        Span::styled("d", Style::default().fg(NEON_YELLOW)),
        Span::styled("] delete  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("c", Style::default().fg(NEON_CYAN)),
        Span::styled(
            format!("] {}  ", config_hint),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("[]", Style::default().fg(NEON_YELLOW)),
        Span::styled("] model ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}/{}", app.selected_model + 1, app.models.len()),
            Style::default().fg(NEON_YELLOW),
        ),
    ]);
    frame.render_widget(Paragraph::new(footer), chunks[2]);
}

fn render_model_cards_grid(app: &App, frame: &mut Frame, area: Rect, cards: &[Card]) {
    if cards.is_empty() {
        frame.render_widget(Paragraph::new("No evaluations"), area);
        return;
    }

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

        if row < scroll {
            continue;
        }

        let visible_row = row - scroll;
        let x = area.x + (col as u16) * card_width;
        let y = area.y + (visible_row as u16) * card_height;

        if y + card_height > area.bottom() {
            continue;
        }

        let card_area = Rect::new(x, y, card_width.min(area.right() - x), card_height);
        let is_selected = i == app.selected_card;

        if let Card::Evaluation { name } = card
            && let Some(eval) = app.get_model_evaluation(name)
        {
            render_evaluation_card(frame, card_area, eval, is_selected);
        }
    }
}

fn render_cards_grid(app: &App, frame: &mut Frame, area: Rect, cards: &[Card]) {
    let Some(run) = app.current_run() else { return };

    if cards.is_empty() {
        frame.render_widget(Paragraph::new("No data"), area);
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
                    render_chart(frame, card_area, name, points, is_selected);
                }
            }
            Card::Examples { name } => {
                if let Some(examples) = run.examples.get(name) {
                    render_examples_card(frame, card_area, name, examples, is_selected);
                }
            }
            Card::Evaluation { .. } => {}
        }
    }
}

fn render_config_panel(frame: &mut Frame, area: Rect, config: &serde_json::Value) {
    let mut lines: Vec<Line> = Vec::new();
    render_json_value(config, 0, &mut lines);

    let block = Block::default()
        .title(Span::styled(
            " CONFIG ",
            Style::default().fg(NEON_CYAN).bold(),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(ratatui::widgets::Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_json_value(value: &serde_json::Value, indent: usize, lines: &mut Vec<Line>) {
    let pad = "  ".repeat(indent);
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                match val {
                    serde_json::Value::Object(_) => {
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}:", key), Style::default().fg(NEON_MAGENTA)),
                        ]));
                        render_json_value(val, indent + 1, lines);
                    }
                    serde_json::Value::Array(arr) => {
                        lines.push(Line::from(vec![
                            Span::styled(pad.clone(), Style::default()),
                            Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                            Span::styled(
                                format!("[{}]", arr.len()),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
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
        _ => {
            let val_str = format_json_primitive(value);
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

fn render_chart(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    points: &[MetricPoint],
    selected: bool,
) {
    use ratatui::symbols::Marker;
    use ratatui::widgets::{Axis, Chart, Dataset, GraphType};

    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    if points.is_empty() {
        let block = Block::default()
            .title(Span::styled(title, title_style))
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

    // Convert points to (x, y) tuples for ratatui
    let data: Vec<(f64, f64)> = points.iter().map(|p| (p.step as f64, p.value)).collect();

    // Find bounds
    let x_min = data.first().map(|p| p.0).unwrap_or(0.0);
    let x_max = data.last().map(|p| p.0).unwrap_or(1.0);
    let y_min = data.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    // Add some padding to y bounds
    let y_range = (y_max - y_min).max(0.001);
    let y_min = y_min - y_range * 0.1;
    let y_max = y_max + y_range * 0.1;

    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(NEON_GREEN))
        .data(&data);

    let axis_style = Style::default().fg(DIM_CYAN);
    let label_style = Style::default().fg(Color::DarkGray);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title(Span::styled(title, title_style))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color)),
        )
        .x_axis(
            Axis::default()
                .style(axis_style)
                .bounds([x_min, x_max])
                .labels(vec![
                    Span::styled(format!("{:.0}", x_min), label_style),
                    Span::styled(format!("{:.0}", x_max), label_style),
                ]),
        )
        .y_axis(
            Axis::default()
                .style(axis_style)
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::styled(format!("{:.2}", y_min), label_style),
                    Span::styled(format!("{:.2}", y_max), label_style),
                ]),
        );

    frame.render_widget(chart, area);
}

fn render_examples_card(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    examples: &[Example],
    selected: bool,
) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    // Show the latest example as preview
    let content: Vec<Line> = if let Some(example) = examples.last() {
        let max_lines = area.height.saturating_sub(4) as usize;
        // Use the first prompt for preview
        let first_prompt = example.prompts.first().map(|s| s.as_str()).unwrap_or("");
        let prompt_preview: String = first_prompt.chars().take(50).collect();
        // Use the first response of first prompt for preview
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

        let mut lines = vec![
            Line::from(vec![
                Span::styled("Q: ", Style::default().fg(NEON_YELLOW).bold()),
                Span::styled(
                    format!("{}...", prompt_preview),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled("A: ", Style::default().fg(NEON_GREEN).bold())),
        ];
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

    // Calculate average reward if rewards exist
    let avg_reward = compute_average_reward(examples);

    // Build title with batch info if applicable
    let batch_size = examples.last().map(|e| e.prompts.len()).unwrap_or(0);
    let mut title_spans = vec![Span::styled(format!("{} ", name), title_style)];

    if batch_size > 1 {
        title_spans.push(Span::styled(
            format!("({} total, batch={})", examples.len(), batch_size),
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        title_spans.push(Span::styled(
            format!("({} total)", examples.len()),
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

fn render_evaluation_card(frame: &mut Frame, area: Rect, eval: &Evaluation, selected: bool) {
    let border_color = if selected { NEON_CYAN } else { DIM_CYAN };
    let title_style = if selected {
        Style::default().fg(NEON_CYAN).bold()
    } else {
        Style::default().fg(NEON_GREEN)
    };

    // Build content: metrics as key-value pairs
    let mut content: Vec<Line> = Vec::new();

    if let Some(metrics) = &eval.metrics {
        let mut keys: Vec<&String> = metrics.keys().collect();
        keys.sort();
        for key in keys.iter().take(area.height.saturating_sub(4) as usize) {
            if let Some(value) = metrics.get(*key) {
                let val_str = match value {
                    serde_json::Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            format!("{:.4}", f)
                        } else {
                            n.to_string()
                        }
                    }
                    _ => value.to_string(),
                };
                content.push(Line::from(vec![
                    Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                    Span::styled(val_str, Style::default().fg(Color::White)),
                ]));
            }
        }
    }

    // Add example count
    if !eval.examples.is_empty() {
        content.push(Line::from(""));
        content.push(Line::from(Span::styled(
            format!("{} examples", eval.examples.len()),
            Style::default().fg(Color::DarkGray),
        )));
    }

    if content.is_empty() {
        content.push(Line::from(Span::styled(
            "No metrics",
            Style::default().fg(Color::DarkGray),
        )));
    }

    // Build title with time info
    let mut title_spans = vec![
        Span::styled("◆ ", Style::default().fg(NEON_YELLOW)),
        Span::styled(&eval.name, title_style),
    ];
    // Show started_at time if available, otherwise logged_at
    if let Some(started_at) = eval.started_at {
        title_spans.push(Span::styled(
            format!("  {}", started_at.format("%m-%d %H:%M")),
            Style::default().fg(Color::DarkGray),
        ));
    } else if let Some(logged_at) = eval.logged_at {
        title_spans.push(Span::styled(
            format!("  {}", logged_at.format("%m-%d %H:%M")),
            Style::default().fg(Color::DarkGray),
        ));
    }
    // Show duration if we have both start and end times
    if let (Some(started), Some(finished)) = (eval.started_at, eval.finished_at) {
        let duration = finished.signed_duration_since(started);
        let duration_str = if duration.num_seconds() < 60 {
            format!("{}s", duration.num_seconds())
        } else if duration.num_minutes() < 60 {
            format!(
                "{}m{}s",
                duration.num_minutes(),
                duration.num_seconds() % 60
            )
        } else {
            format!("{}h{}m", duration.num_hours(), duration.num_minutes() % 60)
        };
        title_spans.push(Span::styled(
            format!("  ({})", duration_str),
            Style::default().fg(NEON_GREEN),
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

fn compute_average_reward(examples: &[Example]) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0;
    for example in examples {
        if let Some(rewards) = &example.rewards {
            for prompt_rewards in rewards {
                for reward in prompt_rewards {
                    total += reward.total();
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        Some(total / count as f64)
    } else {
        None
    }
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

    match app.view_mode {
        ViewMode::Runs => render_focused_run(app, frame, area),
        ViewMode::Models => render_focused_model(app, frame, area),
        ViewMode::Infra => {}
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
                render_chart(frame, chunks[0], name, points, true);
            }
            let footer = Line::from(vec![
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
            ]);
            frame.render_widget(Paragraph::new(footer), chunks[1]);
        }
        Card::Examples { name } => {
            if let Some(examples) = run.examples.get(name) {
                if let Some(example) = examples.get(app.selected_example) {
                    render_focused_example(
                        frame,
                        chunks[0],
                        name,
                        app.selected_example,
                        examples.len(),
                        example,
                        app.selected_prompt,
                        app.selected_response,
                        app.focused_section,
                        app.prompt_scroll_offset,
                        app.response_scroll_offset,
                    );
                }
                let prompt_count = examples
                    .get(app.selected_example)
                    .map(|e| e.prompts.len())
                    .unwrap_or(0);
                let response_count = examples
                    .get(app.selected_example)
                    .and_then(|e| e.responses.get(app.selected_prompt))
                    .map(|r| r.len())
                    .unwrap_or(0);
                let focus_label = match app.focused_section {
                    FocusedSection::Prompt => "prompt",
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
                    Span::styled("j/k", Style::default().fg(NEON_CYAN)),
                    Span::styled("] scroll  ", Style::default().fg(Color::DarkGray)),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("Tab", Style::default().fg(NEON_CYAN)),
                    Span::styled("] focus:", Style::default().fg(Color::DarkGray)),
                    Span::styled(focus_label, Style::default().fg(NEON_GREEN)),
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
                    Span::styled("] example ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", app.selected_example + 1, examples.len()),
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
                let footer = Line::from(footer_spans);
                frame.render_widget(Paragraph::new(footer), chunks[1]);
            }
        }
        Card::Evaluation { .. } => {}
    }
}

fn render_focused_model(app: &App, frame: &mut Frame, area: Rect) {
    let cards = app.model_cards();
    let Some(card) = cards.get(app.selected_card) else {
        frame.render_widget(Paragraph::new("No card selected"), area);
        return;
    };

    let model = app.current_model();
    let model_config = model.and_then(|m| m.config.clone());

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(area);

    // Breadcrumbs: project / model / evaluation (if viewing evaluation)
    if let Some(m) = model {
        let mut breadcrumb_spans = vec![
            Span::styled(&m.project, Style::default().fg(Color::DarkGray)),
            Span::styled(" / ", Style::default().fg(Color::DarkGray)),
            Span::styled(&m.name, Style::default().fg(NEON_CYAN).bold()),
        ];
        if let Card::Evaluation { name } = card {
            breadcrumb_spans.extend(vec![
                Span::styled(" / ", Style::default().fg(Color::DarkGray)),
                Span::styled(name, Style::default().fg(NEON_YELLOW).bold()),
            ]);
        }
        let breadcrumb = Line::from(breadcrumb_spans);
        frame.render_widget(Paragraph::new(breadcrumb), chunks[0]);
    }

    if let Card::Evaluation { name } = card
        && let Some(eval) = app.get_model_evaluation(name)
    {
        render_focused_evaluation(
            frame,
            chunks[1],
            eval,
            app.selected_example,
            app.selected_prompt,
            app.selected_response,
            app.focused_section,
            app.prompt_scroll_offset,
            app.response_scroll_offset,
            app.show_config,
            model_config.as_ref(),
        );
        let example_count = eval.examples.len();
        let current_example = eval.examples.get(app.selected_example);
        let prompt_count = current_example.map(|e| e.prompts.len()).unwrap_or(0);
        let response_count = current_example
            .and_then(|e| e.responses.get(app.selected_prompt))
            .map(|r| r.len())
            .unwrap_or(0);
        let focus_label = match app.focused_section {
            FocusedSection::Prompt => "prompt",
            FocusedSection::Response => "response",
        };
        let config_hint = if app.show_config { "hide" } else { "config" };
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
            Span::styled("c", Style::default().fg(NEON_CYAN)),
            Span::styled(
                format!("] {}  ", config_hint),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("j/k", Style::default().fg(NEON_CYAN)),
            Span::styled("] scroll  ", Style::default().fg(Color::DarkGray)),
        ];
        if example_count > 0 {
            footer_spans.extend(vec![
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("Tab", Style::default().fg(NEON_CYAN)),
                Span::styled("] focus:", Style::default().fg(Color::DarkGray)),
                Span::styled(focus_label, Style::default().fg(NEON_GREEN)),
                Span::styled("  ", Style::default()),
                Span::styled("[", Style::default().fg(DIM_CYAN)),
                Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
                Span::styled("] example ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/{}", app.selected_example + 1, example_count),
                    Style::default().fg(NEON_YELLOW),
                ),
            ]);
            if prompt_count > 1 {
                footer_spans.extend(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("[]", Style::default().fg(NEON_CYAN)),
                    Span::styled("] prompt ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", app.selected_prompt + 1, prompt_count),
                        Style::default().fg(NEON_MAGENTA),
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
        }
        let footer = Line::from(footer_spans);
        frame.render_widget(Paragraph::new(footer), chunks[2]);
    }
}

#[allow(clippy::too_many_arguments)]
fn render_focused_evaluation(
    frame: &mut Frame,
    area: Rect,
    eval: &Evaluation,
    selected_example: usize,
    selected_prompt: usize,
    selected_response: usize,
    focused_section: FocusedSection,
    prompt_scroll_offset: usize,
    response_scroll_offset: usize,
    show_model_config: bool,
    model_config: Option<&serde_json::Value>,
) {
    let (main_area, config_area) = if show_model_config && model_config.is_some() {
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(area);
        (h_chunks[0], Some(h_chunks[1]))
    } else {
        (area, None)
    };

    if let Some(config_area) = config_area {
        let config_content: Vec<Line> = if let Some(config) = model_config {
            let mut lines = Vec::new();
            render_json_value(config, 0, &mut lines);
            lines
        } else {
            vec![Line::from(Span::styled(
                "No config",
                Style::default().fg(Color::DarkGray),
            ))]
        };
        let config_block = Block::default()
            .title(Span::styled(
                "◆ MODEL CONFIG ",
                Style::default().fg(NEON_YELLOW).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(config_content)
                .block(config_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            config_area,
        );
    }

    if eval.examples.is_empty() {
        // No examples, just show metrics and eval config
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_area);

        // Eval config panel (top) - includes timing info
        let mut config_content: Vec<Line> = Vec::new();

        // Add timing info
        if let Some(started) = eval.started_at {
            config_content.push(Line::from(vec![
                Span::styled("started: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    started.format("%Y-%m-%d %H:%M:%S").to_string(),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        if let Some(finished) = eval.finished_at {
            config_content.push(Line::from(vec![
                Span::styled("finished: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    finished.format("%Y-%m-%d %H:%M:%S").to_string(),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        if let (Some(started), Some(finished)) = (eval.started_at, eval.finished_at) {
            let duration = finished.signed_duration_since(started);
            let duration_str = format!(
                "{}.{:03}s",
                duration.num_seconds(),
                duration.num_milliseconds() % 1000
            );
            config_content.push(Line::from(vec![
                Span::styled("duration: ", Style::default().fg(Color::DarkGray)),
                Span::styled(duration_str, Style::default().fg(NEON_GREEN)),
            ]));
        }

        // Add separator if we have timing info and config
        if !config_content.is_empty() && eval.config.is_some() {
            config_content.push(Line::from(""));
        }

        // Add config
        if let Some(config) = &eval.config {
            render_json_value(config, 0, &mut config_content);
        } else if config_content.is_empty() {
            config_content.push(Line::from(Span::styled(
                "No config",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let config_block = Block::default()
            .title(Span::styled(
                "◆ EVAL CONFIG ",
                Style::default().fg(NEON_CYAN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(config_content)
                .block(config_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            chunks[0],
        );

        // Metrics panel (bottom)
        let metrics_content: Vec<Line> = if let Some(metrics) = &eval.metrics {
            let mut lines = Vec::new();
            let mut keys: Vec<&String> = metrics.keys().collect();
            keys.sort();
            for key in keys {
                if let Some(value) = metrics.get(key) {
                    let val_str = match value {
                        serde_json::Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                format!("{:.6}", f)
                            } else {
                                n.to_string()
                            }
                        }
                        _ => value.to_string(),
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                        Span::styled(val_str, Style::default().fg(Color::White)),
                    ]));
                }
            }
            lines
        } else {
            vec![Line::from(Span::styled(
                "No metrics",
                Style::default().fg(Color::DarkGray),
            ))]
        };
        let metrics_block = Block::default()
            .title(Span::styled(
                "◆ METRICS ",
                Style::default().fg(NEON_GREEN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(metrics_content)
                .block(metrics_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            chunks[1],
        );
    } else {
        // Has examples: left side (config + metrics), right side (examples)
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(main_area);

        // Left side: eval config (top) and metrics (bottom)
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(h_chunks[0]);

        // Eval config panel (top left) - includes timing info
        let mut config_content: Vec<Line> = Vec::new();

        // Add timing info
        if let Some(started) = eval.started_at {
            config_content.push(Line::from(vec![
                Span::styled("started: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    started.format("%Y-%m-%d %H:%M:%S").to_string(),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        if let Some(finished) = eval.finished_at {
            config_content.push(Line::from(vec![
                Span::styled("finished: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    finished.format("%Y-%m-%d %H:%M:%S").to_string(),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        if let (Some(started), Some(finished)) = (eval.started_at, eval.finished_at) {
            let duration = finished.signed_duration_since(started);
            let duration_str = format!(
                "{}.{:03}s",
                duration.num_seconds(),
                duration.num_milliseconds() % 1000
            );
            config_content.push(Line::from(vec![
                Span::styled("duration: ", Style::default().fg(Color::DarkGray)),
                Span::styled(duration_str, Style::default().fg(NEON_GREEN)),
            ]));
        }

        // Add separator if we have timing info and config
        if !config_content.is_empty() && eval.config.is_some() {
            config_content.push(Line::from(""));
        }

        // Add config
        if let Some(config) = &eval.config {
            render_json_value(config, 0, &mut config_content);
        } else if config_content.is_empty() {
            config_content.push(Line::from(Span::styled(
                "No config",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let config_block = Block::default()
            .title(Span::styled(
                "◆ EVAL CONFIG ",
                Style::default().fg(NEON_CYAN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(config_content)
                .block(config_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            left_chunks[0],
        );

        // Metrics panel (bottom left)
        let metrics_content: Vec<Line> = if let Some(metrics) = &eval.metrics {
            let mut lines = Vec::new();
            let mut keys: Vec<&String> = metrics.keys().collect();
            keys.sort();
            for key in keys {
                if let Some(value) = metrics.get(key) {
                    let val_str = match value {
                        serde_json::Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                format!("{:.6}", f)
                            } else {
                                n.to_string()
                            }
                        }
                        _ => value.to_string(),
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!("{}: ", key), Style::default().fg(NEON_MAGENTA)),
                        Span::styled(val_str, Style::default().fg(Color::White)),
                    ]));
                }
            }
            lines
        } else {
            vec![Line::from(Span::styled(
                "No metrics",
                Style::default().fg(Color::DarkGray),
            ))]
        };
        let metrics_block = Block::default()
            .title(Span::styled(
                "◆ METRICS ",
                Style::default().fg(NEON_GREEN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(metrics_content)
                .block(metrics_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            left_chunks[1],
        );

        // Right side: Examples card containing prompt and response
        let examples_block = Block::default()
            .title(Span::styled(
                format!(
                    "◆ EXAMPLES #{}/{} ",
                    selected_example + 1,
                    eval.examples.len()
                ),
                Style::default().fg(NEON_YELLOW).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        let examples_inner = examples_block.inner(h_chunks[1]);
        frame.render_widget(examples_block, h_chunks[1]);

        // Split examples inner area into prompt and response
        let example_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(examples_inner);

        // Get current example
        if let Some(example) = eval.examples.get(selected_example) {
            // Get the prompt text
            let prompt_text = example
                .prompts
                .get(selected_prompt)
                .cloned()
                .unwrap_or_default();

            // Prompt title - show batch position if multiple prompts
            let prompt_focused = focused_section == FocusedSection::Prompt;
            let prompt_border_color = if prompt_focused { NEON_CYAN } else { DIM_CYAN };
            let mut prompt_title_spans = vec![Span::styled(
                "PROMPT ",
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
            let prompt_title = Line::from(prompt_title_spans);

            let prompt_block = Block::default()
                .title(prompt_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(prompt_border_color));
            frame.render_widget(
                Paragraph::new(prompt_text)
                    .style(Style::default().fg(Color::White))
                    .block(prompt_block)
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .scroll((prompt_scroll_offset as u16, 0)),
                example_chunks[0],
            );

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
                "RESPONSE ",
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
                                format!("{:.1}", value),
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

            let response_block = Block::default()
                .title(response_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(response_border_color));
            frame.render_widget(
                Paragraph::new(response_text)
                    .style(Style::default().fg(Color::Gray))
                    .block(response_block)
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .scroll((response_scroll_offset as u16, 0)),
                example_chunks[1],
            );
        }
    }
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
) {
    // Layout: prompt and response
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

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
        Span::styled(
            format!(" #{}", index + 1),
            Style::default().fg(Color::White),
        ),
        Span::styled(format!("/{}", total), Style::default().fg(Color::DarkGray)),
        Span::styled(" │ ", Style::default().fg(DIM_CYAN)),
        Span::styled("step ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", example.step), Style::default().fg(NEON_CYAN)),
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
                        format!("{:.1}", value),
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
    frame.render_widget(response, chunks[1]);
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
    } else if let Some(idx) = app.pending_delete_model {
        let name = app
            .models
            .get(idx)
            .map(|m| m.name.as_str())
            .unwrap_or("unknown");
        ("model", name)
    } else if let Some(idx) = app.pending_delete_eval {
        let name = app
            .model_evaluations
            .get(idx)
            .map(|e| e.name.as_str())
            .unwrap_or("unknown");
        ("evaluation", name)
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

fn render_infra_dashboard(app: &App, frame: &mut Frame) {
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
}

fn render_infra_provider_tabs(app: &App, frame: &mut Frame, area: Rect) {
    let lambda_configured = app.infra_config.lambda_config.api_key.is_some();
    let vast_configured = app.infra_config.vast.api_key.is_some();
    let prime_configured = app.infra_config.prime.api_key.is_some();

    let lambda_selected = app.selected_infra_provider == Provider::Lambda;
    let vast_selected = app.selected_infra_provider == Provider::Vast;
    let prime_selected = app.selected_infra_provider == Provider::Prime;

    let spans = vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Runs", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Models", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("[Infra]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled("   Provider: ", Style::default().fg(Color::DarkGray)),
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
        Span::styled("          ", Style::default()),
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
        let empty_text = vec![
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
        ];
        let empty = Paragraph::new(empty_text).block(block);
        frame.render_widget(empty, area);
        return;
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

            let line2 = Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    instance.status.as_str().to_string(),
                    Style::default().fg(status_color),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(&instance.region, Style::default().fg(Color::DarkGray)),
                Span::styled("  ", Style::default()),
                Span::styled(ip_display, Style::default().fg(NEON_CYAN)),
            ]);

            ListItem::new(vec![line1, line2])
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

fn render_infra_types_panel(app: &App, frame: &mut Frame, area: Rect) {
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
        let config = app.infra_config.get_provider_config(provider);
        let msg = if config.api_key.is_none() {
            format!(
                "No API key for {}. Press [c] to configure.",
                provider.display_name()
            )
        } else {
            "No instance types available".to_string()
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

    let mut state = ListState::default();
    if is_focused {
        state.select(Some(app.selected_infra_type));
    }

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_infra_help_bar(app: &App, frame: &mut Frame, area: Rect) {
    let help = if app.launch_selecting_region || app.launch_confirming || app.session_modal_open {
        Line::from(vec![])
    } else if app.infra_active_panel == InfraPanel::Instances {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Tab", Style::default().fg(NEON_CYAN)),
            Span::styled("] panel  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
            Span::styled("] nav  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] ssh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("S", Style::default().fg(NEON_YELLOW)),
            Span::styled("] setup+ssh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("x", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] terminate  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("1-3", Style::default().fg(NEON_YELLOW)),
            Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("R", Style::default().fg(NEON_CYAN)),
            Span::styled("] refresh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] quit", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Tab", Style::default().fg(NEON_CYAN)),
            Span::styled("] panel  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
            Span::styled("] nav  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("] launch  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("s", Style::default().fg(NEON_CYAN)),
            Span::styled("] sort  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("1-3", Style::default().fg(NEON_YELLOW)),
            Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("R", Style::default().fg(NEON_CYAN)),
            Span::styled("] refresh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] quit", Style::default().fg(Color::DarkGray)),
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
    let popup_width = 40u16.min(area.width.saturating_sub(4));
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
        lines.push(Line::from(Span::styled(
            format!("{}{}", prefix, region),
            style,
        )));
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
    let type_price = selected_type
        .map(|t| t.price_display())
        .unwrap_or_else(|| "?".to_string());
    let region = selected_type
        .and_then(|t| t.regions.get(app.launch_selected_region))
        .map(|s| s.as_str())
        .unwrap_or("default");

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
            "  (runs from ~/project on remote)",
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

fn render_infra_config(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let providers = Provider::all();

    let items: Vec<ListItem> = providers
        .iter()
        .enumerate()
        .map(|(i, provider)| {
            let is_selected = i == app.config_provider_index;
            let config = app.infra_config.get_provider_config(*provider);
            let has_key = config.api_key.is_some();

            let (status_icon, status_color) = if has_key {
                ("✓", NEON_GREEN)
            } else {
                ("✗", Color::DarkGray)
            };

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else {
                Style::default().fg(Color::Gray)
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
            Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
            Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("Enter", Style::default().fg(NEON_GREEN)),
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::styled("e", Style::default().fg(NEON_GREEN)),
            Span::styled("] edit key  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("d", Style::default().fg(NEON_YELLOW)),
            Span::styled("] delete key  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q", Style::default().fg(NEON_MAGENTA)),
            Span::styled("/", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] back", Style::default().fg(Color::DarkGray)),
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
            Span::styled("d", Style::default().fg(NEON_MAGENTA)),
            Span::styled("] Clear  ", Style::default().fg(Color::DarkGray)),
            Span::styled("[", Style::default().fg(DIM_CYAN)),
            Span::styled("q/Esc", Style::default().fg(Color::Gray)),
            Span::styled("] Back", Style::default().fg(Color::DarkGray)),
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
