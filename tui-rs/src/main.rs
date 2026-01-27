use std::collections::HashSet;
use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
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
use data::{
    Evaluation, Example, MetricPoint, Model, Reward, Run, delete_evaluation, delete_model,
    load_all_evaluations, load_models, load_runs,
};
use infra::{
    InfraConfig, Instance, InstanceStatus, InstanceType, Provider, get_provider, load_config,
};
use remote::RemoteSync;

enum SyncMessage {
    SyncCompleted,
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
    InfraTypes,
}

// Which section is focused in the focused example view
#[derive(Clone, Copy, PartialEq)]
enum FocusedSection {
    Prompt,
    Response,
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
    selected_infra_provider: Option<Provider>,
    selected_infra_instance: usize,
    selected_infra_type: usize,
    infra_loading: bool,
    infra_error: Option<String>,
    show_terminate_confirm: bool,
    pending_terminate_instance: Option<usize>,
}

impl App {
    fn new() -> Self {
        let runs = load_runs();
        let models = load_models();
        let model_evaluations = load_all_evaluations();
        let infra_config = load_config().unwrap_or_default();
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
            selected_infra_provider: None,
            selected_infra_instance: 0,
            selected_infra_type: 0,
            infra_loading: false,
            infra_error: None,
            show_terminate_confirm: false,
            pending_terminate_instance: None,
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

        let providers_to_query: Vec<Provider> = match self.selected_infra_provider {
            Some(p) => vec![p],
            None => Provider::all().to_vec(),
        };

        for provider in providers_to_query {
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
        self.selected_infra_instance = self
            .selected_infra_instance
            .min(self.infra_instances.len().saturating_sub(1));
    }

    fn refresh_infra_types(&mut self) {
        self.infra_loading = true;
        self.infra_error = None;
        self.infra_types.clear();

        let provider = self
            .selected_infra_provider
            .unwrap_or(self.infra_config.default_provider);
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
        self.selected_infra_type = self
            .selected_infra_type
            .min(self.infra_types.len().saturating_sub(1));
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
            std::process::Command::new("osascript")
                .args([
                    "-e",
                    &format!("tell application \"Terminal\" to do script \"{}\"", ssh_cmd),
                ])
                .spawn()?;
        }

        #[cfg(target_os = "linux")]
        {
            std::process::Command::new("x-terminal-emulator")
                .args(["-e", &ssh_cmd])
                .spawn()?;
        }

        Ok(())
    }

    fn filtered_infra_instances(&self) -> Vec<&Instance> {
        match self.selected_infra_provider {
            Some(p) => self
                .infra_instances
                .iter()
                .filter(|i| i.provider == p)
                .collect(),
            None => self.infra_instances.iter().collect(),
        }
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
            View::InfraList | View::InfraTypes => vec![],
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

    fn handle_key(&mut self, code: KeyCode) {
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
                ViewMode::Infra => self.handle_infra_list_key(code),
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
            View::InfraList => self.handle_infra_list_key(code),
            View::InfraTypes => self.handle_infra_types_key(code),
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

    fn handle_infra_list_key(&mut self, code: KeyCode) {
        let instances = self.filtered_infra_instances();
        let instance_count = instances.len();

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
            KeyCode::Up if self.selected_infra_instance > 0 => {
                self.selected_infra_instance -= 1;
            }
            KeyCode::Down if self.selected_infra_instance < instance_count.saturating_sub(1) => {
                self.selected_infra_instance += 1;
            }
            KeyCode::Tab => {
                self.selected_infra_provider = match self.selected_infra_provider {
                    None => Some(Provider::Lambda),
                    Some(Provider::Lambda) => Some(Provider::Vast),
                    Some(Provider::Vast) => Some(Provider::Prime),
                    Some(Provider::Prime) => None,
                };
                self.selected_infra_instance = 0;
            }
            KeyCode::Char('t') => {
                self.view = View::InfraTypes;
                self.refresh_infra_types();
            }
            KeyCode::Char('x') => {
                if instance_count > 0 && self.selected_infra_instance < instance_count {
                    self.pending_terminate_instance = Some(self.selected_infra_instance);
                    self.show_terminate_confirm = true;
                }
            }
            KeyCode::Char('s') => {
                let instances = self.filtered_infra_instances();
                if let Some(instance) = instances.get(self.selected_infra_instance) {
                    if instance.ip.is_some() {
                        let instance_clone = (*instance).clone();
                        let _ = self.launch_ssh(&instance_clone);
                    }
                }
            }
            KeyCode::Char('R') => {
                self.refresh_infra();
            }
            _ => {}
        }
    }

    fn handle_infra_types_key(&mut self, code: KeyCode) {
        let type_count = self.infra_types.len();

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.view = View::InfraList;
            }
            KeyCode::Up if self.selected_infra_type > 0 => {
                self.selected_infra_type -= 1;
            }
            KeyCode::Down if self.selected_infra_type < type_count.saturating_sub(1) => {
                self.selected_infra_type += 1;
            }
            KeyCode::Tab => {
                let old_provider = self.selected_infra_provider;
                self.selected_infra_provider = match self.selected_infra_provider {
                    None => Some(self.infra_config.default_provider),
                    Some(Provider::Lambda) => Some(Provider::Vast),
                    Some(Provider::Vast) => Some(Provider::Prime),
                    Some(Provider::Prime) => Some(Provider::Lambda),
                };
                if self.selected_infra_provider != old_provider {
                    self.selected_infra_type = 0;
                    self.refresh_infra_types();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let options = parse_options()?;

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

        // Draw the UI
        terminal.draw(|frame| render(&app, frame))?;

        // Handle input (with 100ms timeout for responsive feel)
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            // Only handle key press, not release
            if key.kind == KeyEventKind::Press {
                app.handle_key(key.code);
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

struct Options {
    remote_url: Option<String>,
    token: Option<String>,
}

fn parse_options() -> Result<Options> {
    let mut remote_url = None;
    let mut token = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--remote" => {
                remote_url = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--remote requires a URL"))?,
                );
            }
            "--token" => {
                token = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--token requires a value"))?,
                );
            }
            other => {
                return Err(anyhow::anyhow!("Unknown option: {}", other));
            }
        }
    }
    Ok(Options { remote_url, token })
}

fn remote_runs_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ex")
        .join("remote_runs")
}

fn render(app: &App, frame: &mut Frame) {
    match app.view {
        View::List => match app.view_mode {
            ViewMode::Runs => render_runs_list(app, frame),
            ViewMode::Models => render_models_list(app, frame),
            ViewMode::Infra => render_infra_list(app, frame),
        },
        View::RunDetail => render_run_detail(app, frame),
        View::ModelDetail => render_model_detail(app, frame),
        View::Focused => render_focused(app, frame),
        View::InfraList => render_infra_list(app, frame),
        View::InfraTypes => render_infra_types(app, frame),
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

fn render_infra_list(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let instances = app.filtered_infra_instances();

    let items: Vec<ListItem> = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| {
            let is_selected = i == app.selected_infra_instance;

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

            let ip_display = instance.ip.as_deref().unwrap_or("pending...");

            let mut spans = vec![
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(instance.display_name().to_string(), name_style),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("[{}]", instance.provider.display_name()),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(&instance.instance_type, Style::default().fg(NEON_YELLOW)),
                Span::styled("  ", Style::default()),
                Span::styled(&instance.region, Style::default().fg(Color::DarkGray)),
                Span::styled("  ", Style::default()),
                Span::styled(ip_display, Style::default().fg(NEON_CYAN)),
            ];

            if instance.status.is_active() {
                spans.push(Span::styled(
                    format!("  {}", instance.status.as_str()),
                    Style::default().fg(status_color),
                ));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_infra_instance));

    let provider_filter = match app.selected_infra_provider {
        Some(p) => format!("[{}]", p.display_name()),
        None => "[All]".to_string(),
    };

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Runs", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("Models", Style::default().fg(Color::DarkGray)),
        Span::styled(" | ", Style::default().fg(DIM_CYAN)),
        Span::styled("[Infra]", Style::default().fg(NEON_CYAN).bold()),
        Span::styled("  ", Style::default()),
        Span::styled(&provider_filter, Style::default().fg(NEON_YELLOW)),
        Span::styled(
            format!("  ({} instances)", instances.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    if app.infra_loading {
        let loading = Paragraph::new(Span::styled("Loading...", Style::default().fg(NEON_YELLOW)))
            .block(block);
        frame.render_widget(loading, area);
    } else if let Some(error) = &app.infra_error {
        let err_text = vec![
            Line::from(Span::styled("Error:", Style::default().fg(NEON_MAGENTA))),
            Line::from(Span::styled(
                error.clone(),
                Style::default().fg(Color::White),
            )),
        ];
        let err_para = Paragraph::new(err_text).block(block);
        frame.render_widget(err_para, area);
    } else if items.is_empty() {
        let empty = Paragraph::new(Span::styled(
            "No instances found. Press [R] to refresh.",
            Style::default().fg(Color::DarkGray),
        ))
        .block(block);
        frame.render_widget(empty, area);
    } else {
        let list = List::new(items)
            .block(block)
            .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
            .highlight_symbol("▶ ");
        frame.render_stateful_widget(list, area, &mut state);
    }

    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("r", Style::default().fg(NEON_YELLOW)),
        Span::styled("] runs  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("m", Style::default().fg(NEON_YELLOW)),
        Span::styled("] models  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] provider  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("t", Style::default().fg(NEON_CYAN)),
        Span::styled("] types  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("s", Style::default().fg(NEON_GREEN)),
        Span::styled("] ssh  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("x", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] terminate  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("R", Style::default().fg(NEON_CYAN)),
        Span::styled("] refresh  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] quit", Style::default().fg(Color::DarkGray)),
    ]);
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(Paragraph::new(help), help_area);
}

fn render_infra_types(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let items: Vec<ListItem> = app
        .infra_types
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let is_selected = i == app.selected_infra_type;

            let name_style = if is_selected {
                Style::default().fg(NEON_CYAN).bold()
            } else {
                Style::default().fg(Color::Gray)
            };

            let gpu_info = if let Some(gpu_name) = &it.gpu_name {
                format!("{}x {}", it.gpu_count, gpu_name)
            } else {
                format!("{}x GPU", it.gpu_count)
            };

            let regions_display = if it.regions.is_empty() {
                "no availability".to_string()
            } else if it.regions.len() > 3 {
                format!("{} regions", it.regions.len())
            } else {
                it.regions.join(", ")
            };

            let spans = vec![
                Span::styled(format!("{:<20}", it.name), name_style),
                Span::styled(
                    format!("{:<15}", gpu_info),
                    Style::default().fg(NEON_YELLOW),
                ),
                Span::styled(
                    format!("{:>3} vCPU  ", it.vcpus),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:>4}GB RAM  ", it.memory_gib),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:>6}GB disk  ", it.storage_gib),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(it.price_display(), Style::default().fg(NEON_GREEN)),
                Span::styled("  ", Style::default()),
                Span::styled(regions_display, Style::default().fg(Color::DarkGray)),
            ];

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_infra_type));

    let provider = app
        .selected_infra_provider
        .unwrap_or(app.infra_config.default_provider);

    let title = Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled("Instance Types", Style::default().fg(NEON_CYAN).bold()),
        Span::styled("  ", Style::default()),
        Span::styled(
            format!("[{}]", provider.display_name()),
            Style::default().fg(NEON_YELLOW),
        ),
        Span::styled(
            format!("  ({} types)", app.infra_types.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM_CYAN));

    if app.infra_loading {
        let loading = Paragraph::new(Span::styled("Loading...", Style::default().fg(NEON_YELLOW)))
            .block(block);
        frame.render_widget(loading, area);
    } else if let Some(error) = &app.infra_error {
        let err_text = vec![
            Line::from(Span::styled("Error:", Style::default().fg(NEON_MAGENTA))),
            Line::from(Span::styled(
                error.clone(),
                Style::default().fg(Color::White),
            )),
        ];
        let err_para = Paragraph::new(err_text).block(block);
        frame.render_widget(err_para, area);
    } else if items.is_empty() {
        let empty = Paragraph::new(Span::styled(
            "No instance types available",
            Style::default().fg(Color::DarkGray),
        ))
        .block(block);
        frame.render_widget(empty, area);
    } else {
        let list = List::new(items)
            .block(block)
            .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
            .highlight_symbol("▶ ");
        frame.render_stateful_widget(list, area, &mut state);
    }

    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
        Span::styled("] navigate  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab", Style::default().fg(NEON_CYAN)),
        Span::styled("] switch provider  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("q", Style::default().fg(NEON_MAGENTA)),
        Span::styled("/", Style::default().fg(Color::DarkGray)),
        Span::styled("Esc", Style::default().fg(NEON_MAGENTA)),
        Span::styled("] back", Style::default().fg(Color::DarkGray)),
    ]);
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
