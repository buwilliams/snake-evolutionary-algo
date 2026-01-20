# Evolutionary Snake AI

A from-scratch evolutionary algorithm that trains neural networks to play Snake in Rust. No ML frameworks—just pure Rust with `rand`, `serde`, `crossterm`, and `rayon`.

![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Neural Network**: Configurable feed-forward network with ReLU/Tanh activations
- **Genetic Algorithm**: Tournament selection, uniform crossover, Gaussian mutation, elitism
- **Network Growth**: Networks start small and grow organically based on fitness (NEAT-inspired)
- **Raycast Encoding**: Efficient 28-input spatial representation
- **Snake Game**: 8x8 grid with wall collision, self-collision, and energy-based starvation
- **Terminal Visualization**: Watch trained agents play in real-time
- **Parallel Evaluation**: Rayon-powered multi-core training for faster evolution
- **Reproducible**: Seeded RNG for deterministic training and replay

## Quick Start

```bash
# Build
cargo build --release

# Train for 500 generations
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
| `replay-record <agent.json> [delay_ms]` | Watch the record-setting game |
| `generate-config [output.json]` | Generate default config file |

## How It Works

### Neural Network Architecture

With growth enabled, networks start small and evolve:

```
Start:  Input (28) → Hidden (16-24) → Output (4)
Grow:   Input (28) → Hidden (32) → Hidden (24) → Output (4)
Mature: Input (28) → Hidden (64) → Hidden (48) → Hidden (32) → Output (4)
```

### Raycast Input Encoding

The network receives 28 inputs that efficiently encode spatial relationships:

- **8 directions** (N, NE, E, SE, S, SW, W, NW) × **3 values each**:
  - Distance to wall (normalized 0-1, where 1 = adjacent)
  - Distance to food (normalized 0-1, where 1 = adjacent)
  - Distance to own body (normalized 0-1, where 1 = adjacent)
- **4 direction values**: One-hot encoding of current movement direction

This raycast encoding is far more efficient than raw grid encoding (28 vs 68 inputs) and provides the network with direct spatial relationships rather than requiring it to learn grid interpretation.

- **Output**: 4 values (Up, Down, Left, Right) → argmax selects direction
- **Activation**: ReLU on hidden layers, raw logits on output
- **Growth**: Successful agents earn more neurons/layers (see Network Growth below)

### Genetic Algorithm

1. **Evaluate**: Each agent plays multiple solo games
2. **Select**: Tournament selection (k=5)
3. **Crossover**: Uniform crossover (70% rate)
4. **Mutate**: Gaussian noise (20% rate, 0.5 strength)
5. **Elitism**: Top 20 agents copied unchanged

### Energy System

Instead of arbitrary starvation timers, snakes have energy that depletes with movement:

```
energy -= 1 per step
energy += (base - score × decay) per food eaten (minimum floor)
```

Default settings: `starting_energy=100`, `base=75`, `decay=3`, `minimum=30`

| Score | Energy per Food | Effect |
|-------|-----------------|--------|
| 0 | 75 | Learn basics |
| 5 | 60 | Must be efficient |
| 10 | 45 | Must be optimal |
| 15+ | 30 | Survival mode |

This creates natural curriculum learning - success makes the game harder, forcing increasingly sophisticated play.

### Network Growth

Inspired by NEAT, networks can grow organically rather than using fixed architectures:

```
Small networks evolve fast → Successful agents grow → Complexity emerges naturally
```

| Mechanism | Description |
|-----------|-------------|
| **Variable start** | Agents begin with 1 layer, 16-24 neurons |
| **Fitness-gated growth** | Score ≥ 4 triggers chance to add neurons |
| **Plateau detection** | 500 gens without improvement → floor size increases |
| **Competition** | Different sized networks compete; selection finds optimal size |

This mirrors real evolution - simple organisms dominated early Earth because they adapted quickly. Complexity emerged later when it provided advantage.

## Training Results

Training on 8×8 grid with population of 2000 (network growth + raycast encoding):

| Generation | Best Score | Network Size | Notes |
|------------|------------|--------------|-------|
| 0 | 0-3 | 16-24 neurons | Random initialization |
| ~110 | 5 | ~25 avg neurons | Small networks find solutions fast |
| ~198 | 30 | ~65 avg neurons | Raycast encoding breakthrough |
| ~1522 | 49 | ~65 avg neurons | Networks stabilized at efficient size |

**Key insight**: The combination of raycast encoding (spatial relationships) + network growth (start small, grow as needed) + energy system (curriculum learning) produces highly effective agents.

Previous attempts with grid encoding (68 inputs) plateaued at score 6 after 3,800+ generations. Raycast encoding (28 inputs) achieved score 30 in just 198 generations.

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
    "energy_decay_per_score": 3,
    "minimum_energy_per_food": 30
  },
  "network": {
    "input_size": 28,
    "hidden_layers": [24],
    "output_size": 4,
    "activation": "ReLU"
  },
  "evolution": {
    "population_size": 2000,
    "tournament_size": 5,
    "elitism_count": 20,
    "mutation_rate": 0.2,
    "mutation_strength": 0.5,
    "crossover_rate": 0.7
  },
  "training": {
    "generations": 5000,
    "games_per_evaluation": 3,
    "save_interval": 100,
    "seed": null
  },
  "growth": {
    "enabled": true,
    "min_start_neurons": 16,
    "max_start_neurons": 24,
    "start_layers": 1,
    "growth_score_threshold": 4,
    "growth_probability": 0.2,
    "plateau_generations": 500,
    "max_neurons_per_layer": 64,
    "max_hidden_layers": 3
  }
}
```

## Project Structure

```
src/
├── main.rs           # CLI and training loop
├── config.rs         # Hyperparameters and serialization
├── game.rs           # Snake game engine with raycast encoding
├── neural_network.rs # Feed-forward network with mutation and growth
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

## Future Improvements

### Network Architecture

| Improvement | Description |
|-------------|-------------|
| ~~**NEAT**~~ | Partially implemented - networks grow based on fitness |
| **Recurrent layers** | LSTM/GRU for memory of recent moves and planning |
| **Attention mechanism** | Focus on relevant grid regions |

### Training Strategies

| Improvement | Description |
|-------------|-------------|
| **Curriculum learning** | Start on 4x4 grid, gradually increase to 8x8, 16x16 |
| **Novelty search** | Reward behavioral diversity, not just fitness |
| **Coevolution** | Compete agents against each other |
| **Parallel islands** | Multiple populations with occasional migration |

### Game Mechanics

| Improvement | Description |
|-------------|-------------|
| **Wrapped edges** | Snake wraps around instead of wall death (easier learning) |
| **Growing grid** | Start small, expand as score increases |
| **Multiple food** | Several food items visible at once |

### Performance

| Improvement | Description |
|-------------|-------------|
| ~~**Rayon parallelism**~~ | Implemented - ~2x speedup with parallel evaluation |
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
