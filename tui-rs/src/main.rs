use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

mod data;
use data::{load_runs, MetricPoint, Run};

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
            term_width: 80,
            term_height: 24,
        }
    }

    fn update_size(&mut self, width: u16, height: u16) {
        self.term_width = width;
        self.term_height = height;
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
        let total_rows = (card_count + cols - 1) / cols;
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
    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run event loop
    let mut app = App::new();

    while !app.should_quit {
        // Update terminal size
        let size = terminal.size()?;
        app.update_size(size.width, size.height);

        // Draw the UI
        terminal.draw(|frame| render(&app, frame))?;

        // Handle input (with 100ms timeout for responsive feel)
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Only handle key press, not release
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key.code);
                }
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

    // Create list items from runs
    let items: Vec<ListItem> = app
        .runs
        .iter()
        .map(|run| ListItem::new(run.name.clone()))
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_run));

    let list = List::new(items)
        .block(Block::default().title("Runs").borders(Borders::ALL))
        .highlight_style(Style::default().bg(Color::DarkGray).bold())
        .highlight_symbol("> ");

    frame.render_stateful_widget(list, area, &mut state);

    // Help text at bottom
    let help = "[up/down] navigate  [Enter] select  [q] quit";
    let help_area = Rect::new(area.x + 1, area.bottom() - 1, area.width - 2, 1);
    frame.render_widget(
        Paragraph::new(help).style(Style::default().dim()),
        help_area,
    );
}

fn render_detail(app: &App, frame: &mut Frame) {
    let area = frame.area();

    let Some(run) = app.current_run() else {
        frame.render_widget(Paragraph::new("No run selected"), area);
        return;
    };

    // Layout: header, charts area, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Min(10),    // charts
            Constraint::Length(1),  // footer
        ])
        .split(area);

    // Calculate scroll info for header
    let card_width = 40u16;
    let card_height = 12u16;
    let cols = (chunks[1].width / card_width).max(1) as usize;
    let cards = app.cards();
    let total_cards = cards.len();
    let total_rows = (total_cards + cols - 1) / cols;
    let visible_rows = (chunks[1].height / card_height) as usize;
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
    let header = Paragraph::new(format!(
        "Run: {}  ({} charts, {} examples){}",
        run.name,
        run.metrics.len(),
        total_examples,
        scroll_indicator
    ))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, chunks[0]);

    // Cards grid
    render_cards_grid(app, frame, chunks[1], &cards);

    // Footer
    let footer = format!(
        "[q] back  [arrows] select  [Enter] focus  [j/k] scroll  [[] []] run {}/{}",
        app.selected_run + 1,
        app.runs.len()
    );
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().dim()),
        chunks[2],
    );
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

    let total_rows = (cards.len() + cols - 1) / cols;
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

fn render_chart(frame: &mut Frame, area: Rect, title: &str, points: &[MetricPoint], selected: bool) {
    use ratatui::widgets::{Axis, Chart, Dataset, GraphType};
    use ratatui::symbols::Marker;

    let border_color = if selected { Color::Cyan } else { Color::DarkGray };

    if points.is_empty() {
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));
        frame.render_widget(Paragraph::new("No data").block(block), area);
        return;
    }

    // Convert points to (x, y) tuples for ratatui
    let data: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (p.step as f64, p.value))
        .collect();

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
        .style(Style::default().fg(Color::Yellow))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color)),
        )
        .x_axis(
            Axis::default()
                .bounds([x_min, x_max])
                .labels(vec![
                    Span::raw(format!("{:.0}", x_min)),
                    Span::raw(format!("{:.0}", x_max)),
                ]),
        )
        .y_axis(
            Axis::default()
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.2}", y_min)),
                    Span::raw(format!("{:.2}", y_max)),
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
    let border_color = if selected { Color::Cyan } else { Color::DarkGray };

    // Show the latest example as preview
    let content = if let Some(example) = examples.last() {
        let max_lines = area.height.saturating_sub(4) as usize;
        let prompt_preview: String = example
            .prompt
            .chars()
            .take(50)
            .collect::<String>();
        let response_preview: String = example
            .response
            .lines()
            .take(max_lines.saturating_sub(2))
            .collect::<Vec<_>>()
            .join("\n");

        format!("Q: {}...\n\nA: {}", prompt_preview, response_preview)
    } else {
        "No examples".to_string()
    };

    let block = Block::default()
        .title(format!("Examples: {} ({} total)", name, examples.len()))
        .borders(Borders::ALL)
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
                render_chart(frame, chunks[0], name, points, false);
            }
            // Footer for charts
            let footer = format!(
                "[q] back  [left/right] card {}/{}",
                app.selected_card + 1,
                cards.len()
            );
            frame.render_widget(
                Paragraph::new(footer).style(Style::default().dim()),
                chunks[1],
            );
        }
        Card::Examples { name } => {
            if let Some(examples) = run.examples.get(name) {
                if let Some(example) = examples.get(app.selected_example) {
                    render_focused_example(frame, chunks[0], name, app.selected_example, example);
                }
                // Footer for examples
                let footer = format!(
                    "[q] back  [left/right] card {}/{}  [up/down] example {}/{}",
                    app.selected_card + 1,
                    cards.len(),
                    app.selected_example + 1,
                    examples.len()
                );
                frame.render_widget(
                    Paragraph::new(footer).style(Style::default().dim()),
                    chunks[1],
                );
            }
        }
    }
}

fn render_focused_example(
    frame: &mut Frame,
    area: Rect,
    group_name: &str,
    index: usize,
    example: &data::Example,
) {
    // Layout: prompt and response
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Prompt
    let prompt = Paragraph::new(example.prompt.clone())
        .block(
            Block::default()
                .title(format!(
                    "Prompt ({} #{}, step {})",
                    group_name,
                    index + 1,
                    example.step
                ))
                .borders(Borders::ALL),
        )
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(prompt, chunks[0]);

    // Response
    let response = Paragraph::new(example.response.clone())
        .block(Block::default().title("Response").borders(Borders::ALL))
        .wrap(ratatui::widgets::Wrap { trim: false });
    frame.render_widget(response, chunks[1]);
}
