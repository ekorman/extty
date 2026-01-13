use std::io;
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
use data::{MetricPoint, Run, load_runs};

// The views in our app
#[derive(Clone, Copy, PartialEq)]
enum View {
    List,
    Detail,
    Focused,
}

// A card in the detail grid - either a chart or an example group
#[derive(Clone)]
enum Card {
    Chart { name: String },
    Examples { name: String },
}

// All application state lives here
struct App {
    runs: Vec<Run>,
    selected_run: usize,
    selected_card: usize,
    selected_example: usize, // Index within an example group when focused
    scroll_offset: usize,
    view: View,
    should_quit: bool,
    show_config: bool,
    // Terminal dimensions for layout calculations
    term_width: u16,
    term_height: u16,
}

impl App {
    fn new() -> Self {
        let runs = load_runs();
        App {
            runs,
            selected_run: 0,
            selected_card: 0,
            selected_example: 0,
            scroll_offset: 0,
            view: View::List,
            should_quit: false,
            show_config: false,
            term_width: 80,
            term_height: 24,
        }
    }

    fn update_size(&mut self, width: u16, height: u16) {
        self.term_width = width;
        self.term_height = height;
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

        cards
    }

    fn card_count(&self) -> usize {
        self.current_run()
            .map(|r| r.metrics.len() + r.examples.len())
            .unwrap_or(0)
    }

    fn handle_key(&mut self, code: KeyCode) {
        match self.view {
            View::List => self.handle_list_key(code),
            View::Detail => {
                let (visible_rows, cols) = self.grid_layout();
                self.handle_detail_key(code, visible_rows, cols);
            }
            View::Focused => self.handle_focused_key(code),
        }
    }

    fn handle_list_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up if self.selected_run > 0 => self.selected_run -= 1,
            KeyCode::Down if self.selected_run < self.runs.len().saturating_sub(1) => {
                self.selected_run += 1
            }
            KeyCode::Enter if !self.runs.is_empty() => {
                self.selected_card = 0;
                self.view = View::Detail;
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
            KeyCode::Left if self.selected_card > 0 => self.selected_card -= 1,
            KeyCode::Right if self.selected_card < card_count.saturating_sub(1) => {
                self.selected_card += 1
            }
            KeyCode::Up if self.selected_card > 0 => {
                self.selected_card -= 1;
                // Scroll up if we moved above visible area
                let new_row = self.selected_card / cols;
                if new_row < first_visible_row {
                    self.scroll_offset = new_row;
                }
            }
            KeyCode::Down if self.selected_card < card_count.saturating_sub(1) => {
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
            _ => {}
        }
    }

    fn handle_focused_key(&mut self, code: KeyCode) {
        let card_count = self.card_count();
        let cards = self.cards();
        let current_card = cards.get(self.selected_card);

        // Check if we're viewing an examples group
        let example_count = match current_card {
            Some(Card::Examples { name }) => self
                .current_run()
                .and_then(|r| r.examples.get(name))
                .map(|e| e.len())
                .unwrap_or(0),
            _ => 0,
        };

        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.view = View::Detail,
            // Left/Right navigate between cards
            KeyCode::Left if self.selected_card > 0 => {
                self.selected_card -= 1;
                self.selected_example = 0;
            }
            KeyCode::Right if self.selected_card < card_count.saturating_sub(1) => {
                self.selected_card += 1;
                self.selected_example = 0;
            }
            // Up/Down navigate within example groups
            KeyCode::Up if example_count > 0 && self.selected_example > 0 => {
                self.selected_example -= 1;
            }
            KeyCode::Down
                if example_count > 0 && self.selected_example < example_count.saturating_sub(1) =>
            {
                self.selected_example += 1;
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        // Check for any arguments starting with - or --
        for arg in &args[1..] {
            if arg.starts_with('-') {
                anyhow::bail!("Unrecognized flag: {}", arg);
            }
        }
    }

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run event loop
    let mut app = App::new();
    let mut last_list_refresh = Instant::now();
    let mut last_data_refresh = Instant::now();
    let list_refresh_interval = Duration::from_secs(3);
    let data_refresh_interval = Duration::from_millis(500);

    while !app.should_quit {
        // Update terminal size
        let size = terminal.size()?;
        app.update_size(size.width, size.height);

        // Periodic refresh of run list (less frequent)
        if last_list_refresh.elapsed() >= list_refresh_interval {
            app.refresh_runs();
            last_list_refresh = Instant::now();
            last_data_refresh = Instant::now();
        }

        // Faster refresh of current run data when viewing details
        if matches!(app.view, View::Detail | View::Focused)
            && last_data_refresh.elapsed() >= data_refresh_interval
        {
            app.refresh_current_run();
            last_data_refresh = Instant::now();
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

fn render(app: &App, frame: &mut Frame) {
    match app.view {
        View::List => render_list(app, frame),
        View::Detail => render_detail(app, frame),
        View::Focused => render_focused(app, frame),
    }
}

fn render_list(app: &App, frame: &mut Frame) {
    let area = frame.area();

    // Create list items from runs with styling
    let items: Vec<ListItem> = app
        .runs
        .iter()
        .enumerate()
        .map(|(i, run)| {
            let is_running = run.is_running();
            let is_selected = i == app.selected_run;

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

            ListItem::new(Line::from(vec![
                Span::styled(status_icon, Style::default().fg(status_color)),
                Span::styled(run.name.clone(), name_style),
                Span::styled("  ", Style::default()),
                Span::styled(start_str, Style::default().fg(Color::DarkGray)),
                Span::styled(" → ", Style::default().fg(DIM_CYAN)),
                Span::styled(end_str, time_style),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_run));

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
        Span::styled("Enter", Style::default().fg(NEON_CYAN)),
        Span::styled("] select  ", Style::default().fg(Color::DarkGray)),
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
    let header_text = Line::from(vec![
        Span::styled("◆ ", Style::default().fg(NEON_MAGENTA)),
        Span::styled(&run.name, Style::default().fg(NEON_CYAN).bold()),
        Span::styled("  │  ", Style::default().fg(DIM_CYAN)),
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
        Span::styled(&scroll_indicator, Style::default().fg(NEON_MAGENTA)),
    ]);
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
        Span::styled("↑↓←→", Style::default().fg(NEON_CYAN)),
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
    examples: &[data::Example],
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
        let prompt_preview: String = example.prompt.chars().take(50).collect();
        let response_lines: Vec<&str> = example
            .response
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

    let title = Line::from(vec![
        Span::styled(format!("{} ", name), title_style),
        Span::styled(
            format!("({} total)", examples.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

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
                    );
                }
                // Footer for examples
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
                    Span::styled("  ", Style::default()),
                    Span::styled("[", Style::default().fg(DIM_CYAN)),
                    Span::styled("↑↓", Style::default().fg(NEON_CYAN)),
                    Span::styled("] example ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{}/{}", app.selected_example + 1, examples.len()),
                        Style::default().fg(NEON_YELLOW),
                    ),
                ]);
                frame.render_widget(Paragraph::new(footer), chunks[1]);
            }
        }
    }
}

fn render_focused_example(
    frame: &mut Frame,
    area: Rect,
    group_name: &str,
    index: usize,
    total: usize,
    example: &data::Example,
) {
    // Layout: prompt and response
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // Prompt title
    let prompt_title = Line::from(vec![
        Span::styled("◆ PROMPT ", Style::default().fg(NEON_YELLOW).bold()),
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

    let prompt = Paragraph::new(example.prompt.clone())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .title(prompt_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(prompt, chunks[0]);

    // Response title
    let response_title = Line::from(vec![Span::styled(
        "◆ RESPONSE ",
        Style::default().fg(NEON_GREEN).bold(),
    )]);

    let response = Paragraph::new(example.response.clone())
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .title(response_title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(DIM_CYAN)),
        )
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(response, chunks[1]);
}
