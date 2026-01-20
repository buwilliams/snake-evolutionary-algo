mod config;
mod evolution;
mod game;
mod neural_network;
mod visualizer;

use config::Config;
use evolution::{Population, SavedAgent};
use game::{Cell, GameState};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::time::Instant;
use visualizer::{print_benchmark_results, print_training_header, print_training_stats, Visualizer};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => {
            let config_path = args.get(2).map(|s| s.as_str());
            let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("best_agent.json");
            train(config_path, output_path);
        }
        "watch" => {
            if args.len() < 3 {
                eprintln!("Usage: snake-evolutionary-algo watch <agent.json> [delay_ms]");
                return;
            }
            let agent_path = &args[2];
            let delay_ms = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
            watch(agent_path, delay_ms);
        }
        "benchmark" => {
            if args.len() < 3 {
                eprintln!("Usage: snake-evolutionary-algo benchmark <agent.json> [games]");
                return;
            }
            let agent_path = &args[2];
            let games = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
            benchmark(agent_path, games);
        }
        "generate-config" => {
            let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("config.json");
            generate_config(output_path);
        }
        "replay" => {
            if args.len() < 3 {
                eprintln!("Usage: snake-evolutionary-algo replay <agent.json> [max_frames] [seed]");
                return;
            }
            let agent_path = &args[2];
            let max_frames = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
            let seed = args.get(4).and_then(|s| s.parse().ok());
            replay(agent_path, max_frames, seed);
        }
        "continue" => {
            if args.len() < 3 {
                eprintln!("Usage: snake-evolutionary-algo continue <agent.json> [output.json] [generations]");
                return;
            }
            let agent_path = &args[2];
            let output_path = args.get(3).map(|s| s.as_str()).unwrap_or("best_agent.json");
            let generations = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(500);
            continue_training(agent_path, output_path, generations);
        }
        _ => {
            print_usage();
        }
    }
}

fn print_usage() {
    println!("Evolutionary Snake AI\n");
    println!("Usage:");
    println!("  snake-evolutionary-algo train [config.json] [output.json]");
    println!("  snake-evolutionary-algo continue <agent.json> [output.json] [generations]");
    println!("  snake-evolutionary-algo watch <agent.json> [delay_ms]");
    println!("  snake-evolutionary-algo benchmark <agent.json> [games]");
    println!("  snake-evolutionary-algo replay <agent.json> [max_frames] [seed]");
    println!("  snake-evolutionary-algo generate-config [output.json]");
    println!("\nExamples:");
    println!("  cargo run --release -- train");
    println!("  cargo run --release -- train config.json best_agent.json");
    println!("  cargo run --release -- continue best_agent.json continued.json 500");
    println!("  cargo run --release -- watch best_agent.json 100");
    println!("  cargo run --release -- benchmark best_agent.json 1000");
    println!("  cargo run --release -- replay best_agent.json 30");
}

fn train(config_path: Option<&str>, output_path: &str) {
    let config = match config_path {
        Some(path) => match Config::load(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load config from {}: {}", path, e);
                return;
            }
        },
        None => Config::default(),
    };

    if let Err(e) = config.validate() {
        eprintln!("Invalid configuration: {}", e);
        return;
    }

    print_training_header();
    println!("Configuration:");
    println!("  Grid: {}x{}", config.game.grid_width, config.game.grid_height);
    println!("  Population: {}", config.evolution.population_size);
    println!("  Generations: {}", config.training.generations);
    println!("  Games per evaluation: {}", config.training.games_per_evaluation);
    println!("  Hidden layers: {:?}", config.network.hidden_layers);
    println!("  Mutation rate: {}", config.evolution.mutation_rate);
    println!("  Crossover rate: {}", config.evolution.crossover_rate);
    println!();

    let seed = config.training.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    println!("Random seed: {}\n", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut population = Population::new(config.clone(), &mut rng);

    let start_time = Instant::now();

    for gen in 0..config.training.generations {
        let gen_start = Instant::now();

        population.evaluate(seed);
        let stats = population.stats();

        print_training_stats(
            stats.generation,
            stats.best_fitness,
            stats.average_fitness,
            stats.best_score,
        );

        // Save checkpoint
        if gen % config.training.save_interval == 0 || gen == config.training.generations - 1 {
            if let Some(best) = population.best_agent() {
                let saved = SavedAgent::new(best, gen, population.best_score_ever, &config);
                if let Err(e) = saved.save(output_path) {
                    eprintln!("Warning: Failed to save agent: {}", e);
                } else if gen % config.training.save_interval == 0 {
                    println!("  -> Saved checkpoint to {}", output_path);
                }
            }
        }

        // Evolve to next generation (skip on last generation)
        if gen < config.training.generations - 1 {
            population.evolve(&mut rng);
        }

        let gen_elapsed = gen_start.elapsed();
        if gen % 50 == 0 && gen > 0 {
            println!(
                "  -> Gen {} completed in {:.2}s",
                gen,
                gen_elapsed.as_secs_f64()
            );
        }
    }

    let total_elapsed = start_time.elapsed();
    println!("\n{}", "=".repeat(70));
    println!("Training complete!");
    println!(
        "Total time: {:.2}s ({:.2}s per generation)",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / config.training.generations as f64
    );

    if let Some(best) = population.best_agent() {
        println!("Best agent fitness: {:.2}", best.fitness);
        println!("Best score record: {}", population.best_score_ever);
        let saved = SavedAgent::new(best, config.training.generations - 1, population.best_score_ever, &config);
        if let Err(e) = saved.save(output_path) {
            eprintln!("Failed to save final agent: {}", e);
        } else {
            println!("Final agent saved to: {}", output_path);
        }
    }
}

fn watch(agent_path: &str, delay_ms: u64) {
    let saved = match SavedAgent::load(agent_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load agent from {}: {}", agent_path, e);
            return;
        }
    };

    println!("Loaded agent from generation {} with fitness {:.2}", saved.generation, saved.fitness);
    println!("Press Ctrl+C to exit\n");

    let agent = saved.to_agent();
    let visualizer = Visualizer::new(delay_ms);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    loop {
        if let Err(e) = visualizer.watch_agent(&agent, &saved.config, seed) {
            eprintln!("Visualization error: {}", e);
            break;
        }

        println!("\nRestarting in 2 seconds...");
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    let _ = visualizer.cleanup();
}

fn benchmark(agent_path: &str, games: usize) {
    let saved = match SavedAgent::load(agent_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load agent from {}: {}", agent_path, e);
            return;
        }
    };

    println!(
        "Benchmarking agent from generation {} ({} games)",
        saved.generation, games
    );

    let agent = saved.to_agent();
    let mut scores = Vec::with_capacity(games);
    let mut fitnesses = Vec::with_capacity(games);

    let start = Instant::now();

    for i in 0..games {
        // Use 0 as best_score for benchmarking to get raw fitness values
        let result = agent.play_game(&saved.config, i as u64, 0);
        scores.push(result.score);
        fitnesses.push(result.fitness);

        if (i + 1) % 100 == 0 {
            print!("\rProgress: {}/{}", i + 1, games);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let elapsed = start.elapsed();
    println!("\rCompleted in {:.2}s", elapsed.as_secs_f64());

    print_benchmark_results(games, &scores, &fitnesses);
}

fn generate_config(output_path: &str) {
    let config = Config::default();
    match config.save(output_path) {
        Ok(()) => println!("Default configuration saved to: {}", output_path),
        Err(e) => eprintln!("Failed to save config: {}", e),
    }
}

fn replay(agent_path: &str, max_frames: usize, seed_opt: Option<u64>) {
    let saved = match SavedAgent::load(agent_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load agent from {}: {}", agent_path, e);
            return;
        }
    };

    let seed = seed_opt.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("Replaying trained agent (gen {}, fitness {:.2}, seed {})\n", saved.generation, saved.fitness, seed);

    let agent = saved.to_agent();
    let mut game = GameState::new(&saved.config.game, seed);
    let mut frame = 0;

    while !game.game_over && frame < max_frames {
        println!("Step {:3} | Score: {} | Direction: {:?}", frame, game.score, game.direction);
        print_grid(&game);
        println!();

        let input = game.to_network_input();
        let direction = agent.network.decide(&input);
        game.step(direction);
        frame += 1;
    }

    println!("=== {} ===", if game.game_over { "GAME OVER" } else { "TRUNCATED" });
    println!("Final Score: {}, Steps: {}", game.score, game.steps);
}

fn print_grid(game: &GameState) {
    let width = game.width;
    println!("+{}+", "-".repeat(width * 2 + 1));
    for y in 0..game.height {
        print!("|");
        for x in 0..width {
            let c = match game.get_cell(x, y) {
                Cell::Empty => " .",
                Cell::Snake => " O",
                Cell::SnakeHead => " @",
                Cell::Food => " *",
            };
            print!("{}", c);
        }
        println!(" |");
    }
    println!("+{}+", "-".repeat(width * 2 + 1));
}

fn continue_training(agent_path: &str, output_path: &str, generations: usize) {
    let saved = match SavedAgent::load(agent_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load agent from {}: {}", agent_path, e);
            return;
        }
    };

    let mut config = saved.config.clone();
    config.training.generations = generations;

    print_training_header();
    println!("Continuing training from generation {}", saved.generation);
    println!("Starting fitness: {:.2}\n", saved.fitness);
    println!("Configuration:");
    println!("  Grid: {}x{}", config.game.grid_width, config.game.grid_height);
    println!("  Population: {}", config.evolution.population_size);
    println!("  Additional generations: {}", generations);
    println!("  Games per evaluation: {}", config.training.games_per_evaluation);
    println!();

    let seed = config.training.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    println!("Random seed: {}\n", seed);

    let mut rng = StdRng::seed_from_u64(seed);
    let start_gen = saved.generation + 1;
    let mut population = Population::from_agent(saved.to_agent(), start_gen, saved.best_score, config.clone(), &mut rng);

    let start_time = Instant::now();

    for gen in 0..generations {
        let actual_gen = start_gen + gen;
        let gen_start = Instant::now();

        population.evaluate(seed);
        let stats = population.stats();

        print_training_stats(
            stats.generation,
            stats.best_fitness,
            stats.average_fitness,
            stats.best_score,
        );

        // Save checkpoint
        if gen % config.training.save_interval == 0 || gen == generations - 1 {
            if let Some(best) = population.best_agent() {
                let saved = SavedAgent::new(best, actual_gen, population.best_score_ever, &config);
                if let Err(e) = saved.save(output_path) {
                    eprintln!("Warning: Failed to save agent: {}", e);
                } else if gen % config.training.save_interval == 0 {
                    println!("  -> Saved checkpoint to {}", output_path);
                }
            }
        }

        // Evolve to next generation (skip on last generation)
        if gen < generations - 1 {
            population.evolve(&mut rng);
        }

        let gen_elapsed = gen_start.elapsed();
        if (actual_gen) % 50 == 0 && gen > 0 {
            println!(
                "  -> Gen {} completed in {:.2}s",
                actual_gen,
                gen_elapsed.as_secs_f64()
            );
        }
    }

    let total_elapsed = start_time.elapsed();
    println!("\n{}", "=".repeat(70));
    println!("Training complete!");
    println!(
        "Total time: {:.2}s ({:.2}s per generation)",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / generations as f64
    );

    if let Some(best) = population.best_agent() {
        println!("Best agent fitness: {:.2}", best.fitness);
        println!("Best score record: {}", population.best_score_ever);
        let saved = SavedAgent::new(best, start_gen + generations - 1, population.best_score_ever, &config);
        if let Err(e) = saved.save(output_path) {
            eprintln!("Failed to save final agent: {}", e);
        } else {
            println!("Final agent saved to: {}", output_path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evolution::Agent;

    #[test]
    fn test_training_loop_short() {
        // Quick training test with minimal settings
        let mut config = Config::default();
        config.evolution.population_size = 10;
        config.training.generations = 5;
        config.training.games_per_evaluation = 1;

        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut population = Population::new(config.clone(), &mut rng);

        let initial_fitness = {
            population.evaluate(seed);
            population.stats().best_fitness
        };

        for _ in 0..config.training.generations {
            population.evaluate(seed);
            population.evolve(&mut rng);
        }

        let final_fitness = population.stats().best_fitness;

        // Fitness should be positive
        assert!(initial_fitness > 0.0);
        assert!(final_fitness > 0.0);
    }

    #[test]
    fn test_random_agents_all_complete() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..100 {
            let agent = Agent::new(&config.network, &mut rng);
            let result = agent.play_game(&config, i as u64, 0);

            assert!(result.steps > 0, "Agent {} had 0 steps", i);
            assert!(result.fitness > 0.0, "Agent {} had 0 fitness", i);
        }
    }
}
