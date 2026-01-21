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
mod remote;
use data::{Evaluation, Example, MetricPoint, Reward, Run, load_all_evaluations, load_runs};
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

// The views in our app
#[derive(Clone, Copy, PartialEq)]
enum View {
    List,
    Detail,
    Focused,
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

// Represents an item in the hierarchical list view
#[derive(Clone, Debug)]
enum ListEntry {
    Project { name: String },
    Run { run_index: usize },
}

// All application state lives here
struct App {
    runs: Vec<Run>,
    evaluations: Vec<Evaluation>,
    selected_run: usize,
    selected_list_item: usize, // Current position in the flattened list
    expanded_projects: HashSet<String>, // Which projects are expanded
    selected_card: usize,
    selected_example: usize,  // Index within an example group when focused
    selected_prompt: usize,   // Index within prompts batch for an example
    selected_response: usize, // Index within response variants for an example
    scroll_offset: usize,
    focused_section: FocusedSection, // Which section (prompt/response) is focused
    prompt_scroll_offset: usize,     // Scroll offset for prompt in focused view
    response_scroll_offset: usize,   // Scroll offset for response in focused view
    view: View,
    should_quit: bool,
    show_config: bool,
    show_delete_confirm: bool,
    pending_delete_run: Option<usize>, // Index into runs vector of run to delete
    // Terminal dimensions for layout calculations
    term_width: u16,
    term_height: u16,
}

impl App {
    fn new() -> Self {
        let runs = load_runs();
        let evaluations = load_all_evaluations();
        App {
            runs,
            evaluations,
            selected_run: 0,
            selected_list_item: 0,
            expanded_projects: HashSet::new(),
            selected_card: 0,
            selected_example: 0,
            selected_prompt: 0,
            selected_response: 0,
            scroll_offset: 0,
            focused_section: FocusedSection::Response,
            prompt_scroll_offset: 0,
            response_scroll_offset: 0,
            view: View::List,
            should_quit: false,
            show_config: false,
            show_delete_confirm: false,
            pending_delete_run: None,
            term_width: 80,
            term_height: 24,
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

    fn refresh_runs(&mut self) {
        let current_name = self.runs.get(self.selected_run).map(|r| r.name.clone());
        self.runs = load_runs();
        self.evaluations = load_all_evaluations();

        if let Some(name) = current_name {
            if let Some(idx) = self.runs.iter().position(|r| r.name == name) {
                self.selected_run = idx;
            } else {
                self.selected_run = self.selected_run.min(self.runs.len().saturating_sub(1));
            }
        }

        // Ensure selected_list_item is valid
        let entries = self.list_entries();
        self.selected_list_item = self.selected_list_item.min(entries.len().saturating_sub(1));
    }

    fn refresh_current_run(&mut self) {
        if let Some(run) = self.runs.get(self.selected_run) {
            let path = run.path.clone();
            if let Some(updated) = data::reload_run(&path) {
                self.runs[self.selected_run] = updated;
            }
        }
    }

    fn grid_layout(&self) -> (usize, usize) {
        // Returns (visible_rows, cols) for the card grid
        let card_width = 40u16;
        let card_height = 12u16;
        let grid_height = self.term_height.saturating_sub(5); // header + footer
        let cols = (self.term_width / card_width).max(1) as usize;
        let visible_rows = (grid_height / card_height).max(1) as usize;
        (visible_rows, cols)
    }

    fn current_run(&self) -> Option<&Run> {
        self.runs.get(self.selected_run)
    }

    fn cards(&self) -> Vec<Card> {
        let Some(run) = self.current_run() else {
            return vec![];
        };

        let mut cards = Vec::new();

        // Add charts (sorted by name)
        let mut metric_names: Vec<&String> = run.metrics.keys().collect();
        metric_names.sort();
        for name in metric_names {
            cards.push(Card::Chart { name: name.clone() });
        }

        // Add example groups (sorted by name)
        let mut example_names: Vec<&String> = run.examples.keys().collect();
        example_names.sort();
        for name in example_names {
            cards.push(Card::Examples { name: name.clone() });
        }

        // Add evaluations for this run (sorted by name)
        let run_evaluations: Vec<&Evaluation> = self
            .evaluations
            .iter()
            .filter(|e| e.run_name == run.name)
            .collect();
        for eval in run_evaluations {
            cards.push(Card::Evaluation {
                name: eval.name.clone(),
            });
        }

        cards
    }

    fn get_evaluation(&self, name: &str) -> Option<&Evaluation> {
        let run = self.current_run()?;
        self.evaluations
            .iter()
            .find(|e| e.run_name == run.name && e.name == name)
    }

    fn card_count(&self) -> usize {
        let Some(run) = self.current_run() else {
            return 0;
        };
        let eval_count = self
            .evaluations
            .iter()
            .filter(|e| e.run_name == run.name)
            .count();
        run.metrics.len() + run.examples.len() + eval_count
    }

    fn handle_key(&mut self, code: KeyCode) {
        // If delete confirmation is shown, handle that first
        if self.show_delete_confirm {
            self.handle_delete_confirm_key(code);
            return;
        }

        match self.view {
            View::List => self.handle_list_key(code),
            View::Detail => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_detail_key(code, visible_rows, cols);
            }
            View::Focused => self.handle_focused_key(code),
        }
    }

    fn handle_delete_confirm_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                // Confirm deletion
                if let Some(run_idx) = self.pending_delete_run
                    && let Some(run) = self.runs.get(run_idx)
                {
                    let path = run.path.clone();
                    if data::delete_run(&path).is_err() {
                        // Silently ignore deletion errors for now
                    }
                    // Refresh the runs list
                    self.refresh_runs();
                    // Return to List view if we were on Detail
                    if self.view == View::Detail {
                        self.view = View::List;
                    }
                }
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                // Cancel deletion
                self.show_delete_confirm = false;
                self.pending_delete_run = None;
            }
            _ => {}
        }
    }

    fn handle_list_key(&mut self, code: KeyCode) {
        let entries = self.list_entries();
        let entry_count = entries.len();

        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up if self.selected_list_item > 0 => {
                self.selected_list_item -= 1;
            }
            KeyCode::Down if self.selected_list_item < entry_count.saturating_sub(1) => {
                self.selected_list_item += 1;
            }
            KeyCode::Tab => {
                // Toggle expansion of current project
                if let Some(ListEntry::Project { name }) = entries.get(self.selected_list_item) {
                    if self.expanded_projects.contains(name) {
                        self.expanded_projects.remove(name);
                    } else {
                        self.expanded_projects.insert(name.clone());
                    }
                }
            }
            KeyCode::Enter => {
                match entries.get(self.selected_list_item) {
                    Some(ListEntry::Project { name }) => {
                        // Toggle expansion when pressing Enter on a project
                        if self.expanded_projects.contains(name) {
                            self.expanded_projects.remove(name);
                        } else {
                            self.expanded_projects.insert(name.clone());
                        }
                    }
                    Some(ListEntry::Run { run_index }) => {
                        // Open the run detail view
                        self.selected_run = *run_index;
                        self.selected_card = 0;
                        self.view = View::Detail;
                    }
                    None => {}
                }
            }
            KeyCode::Char('d') => {
                // Delete only works on runs, not projects
                if let Some(ListEntry::Run { run_index }) = entries.get(self.selected_list_item) {
                    self.pending_delete_run = Some(*run_index);
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

        // Calculate visible row range
        let first_visible_row = self.scroll_offset;
        let last_visible_row = (self.scroll_offset + visible_rows).saturating_sub(1);

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.view = View::List,
            KeyCode::Left if self.selected_card > 0 => {
                self.selected_card -= 1;
                // Scroll up if we moved above visible area
                let new_row = self.selected_card / cols;
                if new_row < first_visible_row {
                    self.scroll_offset = new_row;
                }
            }
            KeyCode::Right if self.selected_card < card_count.saturating_sub(1) => {
                self.selected_card += 1;
                // Scroll down if we moved below visible area
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
            KeyCode::PageUp | KeyCode::Char('k') => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            KeyCode::PageDown | KeyCode::Char('j') => {
                self.scroll_offset = (self.scroll_offset + 1).min(max_scroll);
            }
            KeyCode::Char('c') => {
                self.show_config = !self.show_config;
            }
            KeyCode::Char('d') if !self.runs.is_empty() => {
                // Show delete confirmation
                self.pending_delete_run = Some(self.selected_run);
                self.show_delete_confirm = true;
            }
            _ => {}
        }
    }

    fn handle_focused_key(&mut self, code: KeyCode) {
        let card_count = self.card_count();
        let cards = self.cards();
        let current_card = cards.get(self.selected_card);

        // Check if we're viewing an examples group or evaluation
        let example_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .map(|e| e.len())
                .unwrap_or(0),
            Some(Card::Evaluation { name }) => self
                .get_evaluation(name)
                .map(|e| e.examples.len())
                .unwrap_or(0),
            _ => 0,
        };

        // Get current prompt count for the selected example
        let prompt_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .and_then(|e| e.get(self.selected_example))
                .map(|ex| ex.prompts.len())
                .unwrap_or(0),
            _ => 0,
        };

        // Get current response count for the selected example and prompt
        let response_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .and_then(|e| e.get(self.selected_example))
                .and_then(|ex| ex.responses.get(self.selected_prompt))
                .map(|r| r.len())
                .unwrap_or(0),
            _ => 0,
        };

        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.view = View::Detail;
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
            // j/k for scrolling the focused section
            KeyCode::Char('j') => match self.focused_section {
                FocusedSection::Prompt => {
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_add(1);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_add(1);
                }
            },
            KeyCode::Char('k') => match self.focused_section {
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
                    self.prompt_scroll_offset = self.prompt_scroll_offset.saturating_add(10);
                }
                FocusedSection::Response => {
                    self.response_scroll_offset = self.response_scroll_offset.saturating_add(10);
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
            // Left/Right navigate between response variants
            KeyCode::Left if response_count > 1 && self.selected_response > 0 => {
                self.selected_response -= 1;
                self.response_scroll_offset = 0;
            }
            KeyCode::Right
                if response_count > 1
                    && self.selected_response < response_count.saturating_sub(1) =>
            {
                self.selected_response += 1;
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
            // Shift+Tab navigate between example groups (cards)
            KeyCode::BackTab if card_count > 1 => {
                // Cycle through cards
                self.selected_card = if self.selected_card == 0 {
                    card_count.saturating_sub(1)
                } else {
                    self.selected_card - 1
                };
                self.selected_example = 0;
                self.selected_prompt = 0;
                self.selected_response = 0;
                self.prompt_scroll_offset = 0;
                self.response_scroll_offset = 0;
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
            if matches!(app.view, View::Detail | View::Focused) {
                app.refresh_current_run();
            } else {
                app.refresh_runs();
            }
        }

        // Periodic refresh of run list for local runs (less frequent)
        if last_list_refresh.elapsed() >= list_refresh_interval {
            app.refresh_runs();
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
        View::List => render_list(app, frame),
        View::Detail => render_detail(app, frame),
        View::Focused => render_focused(app, frame),
    }

    // Render delete confirmation dialog on top if showing
    if app.show_delete_confirm {
        render_delete_confirm(app, frame);
    }
}

fn render_list(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let entries = app.list_entries();

    // Create list items from hierarchical entries
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

    let list = List::new(items)
        .block(
            Block::default()
                .title(Span::styled(
                    " ◆ TRAINING RUNS ",
                    Style::default().fg(NEON_CYAN).bold(),
                ))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .highlight_style(Style::default().bg(Color::Rgb(30, 40, 50)))
        .highlight_symbol("▶ ");

    frame.render_stateful_widget(list, area, &mut state);

    // Help text at bottom with styling
    let help = Line::from(vec![
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
        Span::styled("] navigate  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("Tab/→", Style::default().fg(NEON_CYAN)),
        Span::styled("] expand  ", Style::default().fg(Color::DarkGray)),
        Span::styled("[", Style::default().fg(DIM_CYAN)),
        Span::styled("←", Style::default().fg(NEON_CYAN)),
        Span::styled("] collapse  ", Style::default().fg(Color::DarkGray)),
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

fn render_detail(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    // Layout: header, main content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(10),   // main content
            Constraint::Length(1), // footer
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
            Card::Evaluation { name } => {
                if let Some(eval) = app.get_evaluation(name) {
                    render_evaluation_card(frame, card_area, eval, is_selected);
                }
            }
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

    // Build title
    let title_spans = vec![
        Span::styled("◆ ", Style::default().fg(NEON_YELLOW)),
        Span::styled(&eval.name, title_style),
    ];
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

    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    let cards = app.cards();
    let Some(card) = cards.get(app.selected_card) else {
        frame.render_widget(Paragraph::new("No card selected"), area);
        return;
    };

    // Layout: content and footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    match card {
        Card::Chart { name } => {
            if let Some(points) = run.metrics.get(name) {
                render_chart(frame, chunks[0], name, points, true);
            }
            // Footer for charts
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
                // Get prompt and response counts for footer
                let prompt_count = examples
                    .get(app.selected_example)
                    .map(|e| e.prompts.len())
                    .unwrap_or(0);
                let response_count = examples
                    .get(app.selected_example)
                    .and_then(|e| e.responses.get(app.selected_prompt))
                    .map(|r| r.len())
                    .unwrap_or(0);
                // Footer for examples
                let focus_label = match app.focused_section {
                    FocusedSection::Prompt => "prompt",
                    FocusedSection::Response => "response",
                };
                let mut footer_spans = vec![
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                    Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
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
                // Add prompt navigation if multiple prompts in batch
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
                // Add response variant navigation if multiple responses
                if response_count > 1 {
                    footer_spans.extend(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled("[", Style::default().fg(DIM_CYAN)),
                        Span::styled("←→", Style::default().fg(NEON_CYAN)),
                        Span::styled("] group ", Style::default().fg(Color::DarkGray)),
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
        Card::Evaluation { name } => {
            if let Some(eval) = app.get_evaluation(name) {
                render_focused_evaluation(
                    frame,
                    chunks[0],
                    eval,
                    app.selected_example,
                    app.focused_section,
                    app.prompt_scroll_offset,
                    app.response_scroll_offset,
                );
                // Footer for evaluations
                let example_count = eval.examples.len();
                let focus_label = match app.focused_section {
                    FocusedSection::Prompt => "prompt",
                    FocusedSection::Response => "response",
                };
                let mut footer_spans = vec![
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("q", Style::default().fg(NEON_MAGENTA)),
                    Span::styled("] back  ", Style::default().fg(Color::DarkGray)),
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
                }
                let footer = Line::from(footer_spans);
                frame.render_widget(Paragraph::new(footer), chunks[1]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_focused_evaluation(
    frame: &mut Frame,
    area: Rect,
    eval: &Evaluation,
    selected_example: usize,
    focused_section: FocusedSection,
    prompt_scroll_offset: usize,
    response_scroll_offset: usize,
) {
    if eval.examples.is_empty() {
        // No examples, just show metrics and config
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Config panel (top)
        let config_content: Vec<Line> = if let Some(config) = &eval.config {
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
                "◆ CONFIG ",
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
        // Has examples: show metrics/config at top, prompt/response at bottom
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Percentage(35),
                Constraint::Percentage(50),
            ])
            .split(area);

        // Metrics summary (top)
        let metrics_content: Vec<Line> = if let Some(metrics) = &eval.metrics {
            let mut lines = Vec::new();
            let mut keys: Vec<&String> = metrics.keys().collect();
            keys.sort();
            for key in keys.iter().take(5) {
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
                format!("◆ {} METRICS ", eval.name),
                Style::default().fg(NEON_GREEN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(DIM_CYAN));
        frame.render_widget(
            Paragraph::new(metrics_content)
                .block(metrics_block)
                .wrap(ratatui::widgets::Wrap { trim: false }),
            chunks[0],
        );

        // Get current example
        let example = eval.examples.get(selected_example);
        let prompt_text = example.map(|e| e.prompt.as_str()).unwrap_or("");
        let response_text = example.map(|e| e.response.as_str()).unwrap_or("");

        // Prompt section
        let prompt_focused = focused_section == FocusedSection::Prompt;
        let prompt_border_color = if prompt_focused { NEON_CYAN } else { DIM_CYAN };
        let prompt_block = Block::default()
            .title(Span::styled(
                format!("◆ PROMPT {}/{}", selected_example + 1, eval.examples.len()),
                Style::default().fg(NEON_YELLOW).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(prompt_border_color));
        frame.render_widget(
            Paragraph::new(prompt_text)
                .style(Style::default().fg(Color::White))
                .block(prompt_block)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .scroll((prompt_scroll_offset as u16, 0)),
            chunks[1],
        );

        // Response section
        let response_focused = focused_section == FocusedSection::Response;
        let response_border_color = if response_focused {
            NEON_CYAN
        } else {
            DIM_CYAN
        };
        let response_block = Block::default()
            .title(Span::styled(
                "◆ RESPONSE ",
                Style::default().fg(NEON_GREEN).bold(),
            ))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(response_border_color));
        frame.render_widget(
            Paragraph::new(response_text)
                .style(Style::default().fg(Color::Gray))
                .block(response_block)
                .wrap(ratatui::widgets::Wrap { trim: false })
                .scroll((response_scroll_offset as u16, 0)),
            chunks[2],
        );
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

    // Get the run name to display
    let run_name = app
        .pending_delete_run
        .and_then(|idx| app.runs.get(idx))
        .map(|r| r.name.as_str())
        .unwrap_or("unknown");

    // Create centered popup
    let area = frame.area();
    let popup_width = 60u16.min(area.width.saturating_sub(4));
    let popup_height = 7u16;
    let x = (area.width.saturating_sub(popup_width)) / 2;
    let y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(x, y, popup_width, popup_height);

    // Clear the area behind the popup
    frame.render_widget(Clear, popup_area);

    // Create the popup content
    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Delete run ", Style::default().fg(Color::White)),
            Span::styled(run_name, Style::default().fg(NEON_CYAN).bold()),
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
