# Evolutionary Snake AI

A from-scratch evolutionary algorithm that trains neural networks to play Snake in Rust. No ML frameworks—just pure Rust with `rand`, `serde`, and `crossterm`.

![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Neural Network**: Configurable feed-forward network with ReLU/Tanh activations
- **Genetic Algorithm**: Tournament selection, uniform crossover, Gaussian mutation, elitism
- **Snake Game**: 8x8 grid with wall collision, self-collision, and starvation mechanics
- **Terminal Visualization**: Watch trained agents play in real-time
- **Reproducible**: Seeded RNG for deterministic training and replay

## Quick Start

```bash
# Build
cargo build --release

# Train for 500 generations (takes ~1 minute)
cargo run --release -- train

# Watch the trained agent play
cargo run --release -- watch best_agent.json 100

# Benchmark over 1000 games
cargo run --release -- benchmark best_agent.json 1000
```

## Commands

| Command | Description |
|---------|-------------|
| `train [config.json] [output.json]` | Train from scratch |
| `continue <agent.json> [output.json] [generations]` | Continue training from saved agent |
| `watch <agent.json> [delay_ms]` | Visualize agent playing (default 100ms) |
| `benchmark <agent.json> [games]` | Run performance benchmark |
| `replay <agent.json> [frames] [seed]` | Text-based game replay |
| `generate-config [output.json]` | Generate default config file |

## How It Works

### Neural Network Architecture

```
Input (68) → Hidden (64) → Hidden (32) → Output (4)
```

- **Input**: 8×8 grid (64 cells) + 4 direction one-hot = 68 values
  - Grid encoding: 0.0=empty, 0.33=body, 0.66=head, 1.0=food
- **Output**: 4 values (Up, Down, Left, Right) → argmax selects direction
- **Activation**: ReLU on hidden layers, raw logits on output

### Genetic Algorithm

1. **Evaluate**: Each agent plays 3 games, fitness averaged
2. **Select**: Tournament selection (k=5)
3. **Crossover**: Uniform crossover (70% rate)
4. **Mutate**: Gaussian noise (10% rate, 0.3 strength)
5. **Elitism**: Top 10 agents copied unchanged

### Fitness Function

```
fitness = (score + 1)² × 10 + steps × 0.1
```

Rewards both food collection (quadratic) and survival (linear).

## Training Results

Training on 8×8 grid with population of 500:

| Generation | Avg Fitness | Best Score | Time |
|------------|-------------|------------|------|
| 0 | 13 | 0-1 | - |
| 500 | 22.5 | 2-3 | 55s |
| 1000 | 28 | 2-3 | +141s |

Benchmark (1000 games):
- **Average Score**: 0.27
- **Max Score**: 3
- **Average Fitness**: 21.72

## Configuration

Generate a config file to customize training:

```bash
cargo run --release -- generate-config config.json
```

```json
{
  "game": {
    "grid_width": 8,
    "grid_height": 8,
    "max_steps_without_food": 100
  },
  "network": {
    "input_size": 68,
    "hidden_layers": [64, 32],
    "output_size": 4,
    "activation": "ReLU"
  },
  "evolution": {
    "population_size": 500,
    "tournament_size": 5,
    "elitism_count": 10,
    "mutation_rate": 0.1,
    "mutation_strength": 0.3,
    "crossover_rate": 0.7
  },
  "training": {
    "generations": 500,
    "games_per_evaluation": 3,
    "save_interval": 50,
    "seed": null
  }
}
```

## Project Structure

```
src/
├── main.rs           # CLI and training loop
├── config.rs         # Hyperparameters and serialization
├── game.rs           # Snake game engine
├── neural_network.rs # Feed-forward network with mutation
├── evolution.rs      # Genetic algorithm
└── visualizer.rs     # Terminal rendering
```

## Replay Example

```
Step  12 | Score: 2 | Direction: Up
+-----------------+
| . . . . . . . . |
| . . . @ . . . . |
| . . . O O . . . |
| . . . . . * . . |
| . . . . . . . . |
| . . . . . . . . |
| . . . . . . . . |
| . . . . . . . . |
+-----------------+
```
- `@` = Snake head
- `O` = Snake body
- `*` = Food
- `.` = Empty

## Pre-trained Agents

Two trained agents are included:

- `best_agent.json` - Generation 499 (55s training)
- `best_agent_continued.json` - Generation 999 (196s total training)

## License

MIT
