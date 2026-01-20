use crate::config::Config;
use crate::evolution::Agent;
use crate::game::{Cell, GameState};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType},
};
use std::io::{stdout, Write};
use std::thread;
use std::time::Duration;

pub struct Visualizer {
    delay_ms: u64,
}

impl Visualizer {
    pub fn new(delay_ms: u64) -> Self {
        Self { delay_ms }
    }

    pub fn render(&self, game: &GameState) -> std::io::Result<()> {
        let mut stdout = stdout();

        execute!(stdout, MoveTo(0, 0), Clear(ClearType::All), Hide)?;

        // Top border
        let border_width = game.width * 2 + 3;
        writeln!(stdout, "+{}+", "-".repeat(border_width - 2))?;

        // Grid
        for y in 0..game.height {
            write!(stdout, "| ")?;
            for x in 0..game.width {
                let cell = game.get_cell(x, y);
                let (symbol, color) = match cell {
                    Cell::Empty => (".", Color::DarkGrey),
                    Cell::Snake => ("O", Color::Green),
                    Cell::SnakeHead => ("@", Color::Yellow),
                    Cell::Food => ("*", Color::Red),
                };
                execute!(
                    stdout,
                    SetForegroundColor(color),
                    Print(symbol),
                    Print(" "),
                    ResetColor
                )?;
            }
            writeln!(stdout, "|")?;
        }

        // Bottom border
        writeln!(stdout, "+{}+", "-".repeat(border_width - 2))?;

        // Stats
        writeln!(stdout)?;
        writeln!(
            stdout,
            "Score: {}  Steps: {}  Length: {}",
            game.score,
            game.steps,
            game.snake.len()
        )?;

        if game.game_over {
            execute!(stdout, SetForegroundColor(Color::Red))?;
            writeln!(stdout, "\nGAME OVER!")?;
            execute!(stdout, ResetColor)?;
        }

        stdout.flush()?;
        Ok(())
    }

    pub fn watch_agent(&self, agent: &Agent, config: &Config, seed: u64) -> std::io::Result<()> {
        let mut game = GameState::new(&config.game, seed);
        let mut stdout = stdout();

        execute!(stdout, Hide)?;

        while !game.game_over {
            self.render(&game)?;

            let input = game.to_network_input();
            let direction = agent.network.decide(&input);
            game.step(direction);

            thread::sleep(Duration::from_millis(self.delay_ms));
        }

        // Final render
        self.render(&game)?;

        execute!(stdout, Show)?;

        Ok(())
    }

    pub fn cleanup(&self) -> std::io::Result<()> {
        let mut stdout = stdout();
        execute!(stdout, Show, ResetColor)?;
        Ok(())
    }
}

pub fn print_training_stats(
    generation: usize,
    best_fitness: f64,
    avg_fitness: f64,
    best_score: usize,
) {
    println!(
        "Gen {:4} | Best Fitness: {:8.2} | Avg Fitness: {:8.2} | Best Score: {:3}",
        generation, best_fitness, avg_fitness, best_score
    );
}

pub fn print_training_header() {
    println!("{}", "=".repeat(70));
    println!("              Evolutionary Snake AI - Training");
    println!("{}", "=".repeat(70));
    println!();
}

pub fn print_benchmark_results(games: usize, scores: &[usize], fitnesses: &[f64]) {
    let avg_score: f64 = scores.iter().sum::<usize>() as f64 / games as f64;
    let max_score = scores.iter().max().copied().unwrap_or(0);
    let min_score = scores.iter().min().copied().unwrap_or(0);

    let avg_fitness: f64 = fitnesses.iter().sum::<f64>() / games as f64;
    let max_fitness = fitnesses
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    println!("\n=== Benchmark Results ({} games) ===", games);
    println!("Score:   avg={:.2}, min={}, max={}", avg_score, min_score, max_score);
    println!(
        "Fitness: avg={:.2}, max={:.2}",
        avg_fitness, max_fitness
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_creation() {
        let vis = Visualizer::new(100);
        assert_eq!(vis.delay_ms, 100);
    }

    #[test]
    fn test_stats_formatting() {
        // Just verify these don't panic
        print_training_header();
        print_training_stats(10, 150.5, 75.3, 5);
        print_benchmark_results(10, &[1, 2, 3, 4, 5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
    }
}
