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
| `replay-record <agent.json> [frames]` | Replay the record-setting game |
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
base_fitness = (score + 1)² × 10 + steps × 0.1
```

The fitness is then modified based on competitive pressure:

| Condition | Modifier |
|-----------|----------|
| Beat the record | base × 3 + 500 |
| Tied the record | base × 1.5 |
| Scored 0 (when record > 0) | base × 0.1 |
| Below record | base × (score / record) |

This competitive system prevents "safe looping" behavior where agents survive without hunting food.

### Energy System

Instead of arbitrary starvation timers, snakes have energy that depletes with movement:

```
energy -= 1 per step
energy += (base - score × decay) per food eaten (minimum floor)
```

Default settings: `starting_energy=100`, `base=75`, `decay=5`, `minimum=20`

| Score | Energy per Food | Effect |
|-------|-----------------|--------|
| 0 | 75 | Learn basics |
| 5 | 50 | Must be efficient |
| 10 | 25 | Must be optimal |
| 11+ | 20 | Survival mode |

This creates natural curriculum learning - success makes the game harder, forcing increasingly sophisticated play.

## Training Results

Training on 8×8 grid with population of 500 (energy system + competitive fitness):

| Generation | Best Score | Notes |
|------------|------------|-------|
| 0 | 2 | First record set |
| ~100 | 5 | Early improvement |
| ~1200 | 6 | Extended training |
| 2200 | 6 | Current best |

Benchmark (1000 games):
- **Average Score**: 0.48
- **Max Score**: 6 (training), 3 (benchmark)
- **Average Fitness**: 280.35

The energy system with score-based decay creates emergent difficulty scaling.

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
    "starting_energy": 100,
    "energy_per_food": 75,
    "energy_decay_per_score": 5,
    "minimum_energy_per_food": 20
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

## Future Improvements

The current implementation reaches a practical ceiling around score 8-15. To achieve human-level play (30+), consider these improvements:

### Input Encoding

| Improvement | Description |
|-------------|-------------|
| **Raycast sensors** | Distance to wall/body/food in 8 directions (24 inputs vs 64 grid) |
| **Relative coordinates** | Food position relative to head, not absolute grid |
| **Body density map** | Convolution-friendly representation of nearby danger |
| **Path availability** | Binary flags for which directions are safe 1-3 moves ahead |

### Network Architecture

| Improvement | Description |
|-------------|-------------|
| **Larger networks** | [128, 64, 32] or deeper for complex spatial reasoning |
| **NEAT** | Evolve topology alongside weights (NeuroEvolution of Augmenting Topologies) |
| **Recurrent layers** | LSTM/GRU for memory of recent moves and planning |
| **Attention mechanism** | Focus on relevant grid regions |

### Training Strategies

| Improvement | Description |
|-------------|-------------|
| **Curriculum learning** | Start on 4x4 grid, gradually increase to 8x8, 16x16 |
| **Novelty search** | Reward behavioral diversity, not just fitness |
| **Coevolution** | Compete agents against each other |
| **Parallel islands** | Multiple populations with occasional migration |
| **Adaptive mutation** | Decrease mutation rate as fitness plateaus |

### Game Mechanics

| Improvement | Description |
|-------------|-------------|
| **Wrapped edges** | Snake wraps around instead of wall death (easier learning) |
| **Growing grid** | Start small, expand as score increases |
| **Multiple food** | Several food items visible at once |

### Performance

| Improvement | Description |
|-------------|-------------|
| **Rayon parallelism** | Evaluate agents in parallel across CPU cores |
| **GPU acceleration** | Batch forward passes on GPU |
| **Speciation** | Protect innovative solutions from competition |

### Metrics & Debugging

| Improvement | Description |
|-------------|-------------|
| **Heatmaps** | Visualize where agents die most often |
| **Decision logging** | Record network activations for analysis |
| **Fitness components** | Track score vs survival contributions separately |
| **Generational diversity** | Monitor genetic diversity to detect premature convergence |

## License

MIT
