use crate::config::{Config, NetworkConfig};
use crate::game::GameState;
use crate::neural_network::NeuralNetwork;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameResult {
    pub score: usize,
    pub steps: usize,
    pub fitness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub network: NeuralNetwork,
    #[serde(default)]
    pub fitness: f64,
}

impl Agent {
    pub fn new(network_config: &NetworkConfig, rng: &mut StdRng) -> Self {
        Self {
            network: NeuralNetwork::new(network_config, rng),
            fitness: 0.0,
        }
    }

    pub fn play_game(&self, config: &Config, seed: u64, best_score: usize) -> GameResult {
        let mut game = GameState::new(&config.game, seed);

        while !game.game_over {
            let input = game.to_network_input();
            let direction = self.network.decide(&input);
            game.step(direction);
        }

        let fitness = calculate_fitness(game.score, game.steps, best_score);

        GameResult {
            score: game.score,
            steps: game.steps,
            fitness,
        }
    }

    /// Returns (max_score, seed_of_best_game)
    pub fn evaluate(&mut self, config: &Config, base_seed: u64, best_score: usize) -> (usize, u64) {
        let mut total_fitness = 0.0;
        let mut max_score = 0;
        let mut best_seed = base_seed;

        for i in 0..config.training.games_per_evaluation {
            let seed = base_seed.wrapping_add(i as u64 * 1000);
            let result = self.play_game(config, seed, best_score);
            total_fitness += result.fitness;
            if result.score > max_score {
                max_score = result.score;
                best_seed = seed;
            }
        }

        self.fitness = total_fitness / config.training.games_per_evaluation as f64;
        (max_score, best_seed)
    }
}

pub fn calculate_fitness(score: usize, steps: usize, best_score: usize) -> f64 {
    let base_fitness = ((score + 1) as f64).powi(2) * 10.0 + steps as f64 * 0.1;

    if score > best_score {
        // Beat the record! Big bonus
        base_fitness * 3.0 + 500.0
    } else if score == best_score && best_score > 0 {
        // Tied the record - still good
        base_fitness * 1.5
    } else if score == 0 && best_score > 0 {
        // Scored nothing when record exists - heavy penalty
        base_fitness * 0.1
    } else {
        // Below record but scored something
        base_fitness * (score as f64 / (best_score.max(1) as f64)).max(0.2)
    }
}

#[derive(Debug)]
pub struct PopulationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub best_score: usize,
    pub average_score: f64,
}

pub struct Population {
    pub agents: Vec<Agent>,
    pub generation: usize,
    pub best_score_ever: usize,         // The record to beat
    pub best_score_seed: Option<u64>,   // Seed of the game that set the record
    pub record_agent: Option<Agent>,    // The agent that set the record
    config: Config,
}

impl Population {
    pub fn new(config: Config, rng: &mut StdRng) -> Self {
        let agents = (0..config.evolution.population_size)
            .map(|_| Agent::new(&config.network, rng))
            .collect();

        Self {
            agents,
            generation: 0,
            best_score_ever: 0,
            best_score_seed: None,
            record_agent: None,
            config,
        }
    }

    /// Create a population seeded from an existing agent
    /// The first agent is the original, the rest are mutations of it
    pub fn from_agent(agent: Agent, start_generation: usize, best_score: usize, best_score_seed: Option<u64>, record_agent: Option<Agent>, config: Config, rng: &mut StdRng) -> Self {
        let mut agents = Vec::with_capacity(config.evolution.population_size);

        // Keep the original agent
        agents.push(agent.clone());

        // Fill the rest with mutated copies
        for _ in 1..config.evolution.population_size {
            let mut mutated = agent.clone();
            mutated.network.mutate(
                config.evolution.mutation_rate,
                config.evolution.mutation_strength,
                rng,
            );
            mutated.fitness = 0.0;
            agents.push(mutated);
        }

        Self {
            agents,
            generation: start_generation,
            best_score_ever: best_score,
            best_score_seed,
            record_agent,
            config,
        }
    }

    pub fn evaluate(&mut self, base_seed: u64) {
        let seed_offset = self.generation as u64 * 10000;
        let mut generation_best_score = 0;
        let mut generation_best_seed = 0;
        let mut generation_best_agent_idx = 0;

        for (i, agent) in self.agents.iter_mut().enumerate() {
            let agent_seed = base_seed.wrapping_add(seed_offset).wrapping_add(i as u64);
            let (agent_best, best_game_seed) = agent.evaluate(&self.config, agent_seed, self.best_score_ever);
            if agent_best > generation_best_score {
                generation_best_score = agent_best;
                generation_best_seed = best_game_seed;
                generation_best_agent_idx = i;
            }
        }

        // Update the record if beaten
        if generation_best_score > self.best_score_ever {
            println!("  *** NEW RECORD: {} (was {}) | Seed: {} ***", generation_best_score, self.best_score_ever, generation_best_seed);
            self.best_score_ever = generation_best_score;
            self.best_score_seed = Some(generation_best_seed);
            self.record_agent = Some(self.agents[generation_best_agent_idx].clone());
        }

        // Sort by fitness (descending)
        self.agents
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    pub fn tournament_select(&self, rng: &mut StdRng) -> &Agent {
        let mut best: Option<&Agent> = None;

        for _ in 0..self.config.evolution.tournament_size {
            let idx = rng.gen_range(0..self.agents.len());
            let candidate = &self.agents[idx];

            match best {
                None => best = Some(candidate),
                Some(current_best) => {
                    if candidate.fitness > current_best.fitness {
                        best = Some(candidate);
                    }
                }
            }
        }

        best.unwrap()
    }

    pub fn evolve(&mut self, rng: &mut StdRng) {
        let evolution_config = &self.config.evolution;
        let mut new_agents = Vec::with_capacity(evolution_config.population_size);

        // Elitism: copy top N agents unchanged
        for agent in self.agents.iter().take(evolution_config.elitism_count) {
            new_agents.push(agent.clone());
        }

        // Fill the rest with offspring
        while new_agents.len() < evolution_config.population_size {
            let parent1 = self.tournament_select(rng);
            let parent2 = self.tournament_select(rng);

            let mut child_network = if rng.gen::<f64>() < evolution_config.crossover_rate {
                parent1.network.crossover(&parent2.network, rng)
            } else {
                parent1.network.clone()
            };

            child_network.mutate(
                evolution_config.mutation_rate,
                evolution_config.mutation_strength,
                rng,
            );

            new_agents.push(Agent {
                network: child_network,
                fitness: 0.0,
            });
        }

        self.agents = new_agents;
        self.generation += 1;
    }

    pub fn stats(&self) -> PopulationStats {
        let best_fitness = self.agents.first().map(|a| a.fitness).unwrap_or(0.0);
        let average_fitness: f64 =
            self.agents.iter().map(|a| a.fitness).sum::<f64>() / self.agents.len() as f64;

        PopulationStats {
            generation: self.generation,
            best_fitness,
            average_fitness,
            best_score: self.best_score_ever,
            average_score: estimate_score_from_fitness_f64(average_fitness),
        }
    }

    pub fn best_agent(&self) -> Option<&Agent> {
        self.agents.first()
    }
}

fn estimate_score_from_fitness(fitness: f64) -> usize {
    // Inverse of fitness formula: fitness = (score+1)^2 * 10 + steps * 0.1
    // Approximation assuming steps contribute minimally
    let score_component = fitness / 10.0;
    let estimated_score = score_component.sqrt() - 1.0;
    estimated_score.max(0.0) as usize
}

fn estimate_score_from_fitness_f64(fitness: f64) -> f64 {
    let score_component = fitness / 10.0;
    let estimated_score = score_component.sqrt() - 1.0;
    estimated_score.max(0.0)
}

#[derive(Serialize, Deserialize)]
pub struct SavedAgent {
    pub network: NeuralNetwork,
    pub fitness: f64,
    pub generation: usize,
    #[serde(default)]
    pub best_score: usize,  // The record score at time of save
    #[serde(default)]
    pub best_score_seed: Option<u64>,  // Seed of the game that set the record
    #[serde(default)]
    pub record_network: Option<NeuralNetwork>,  // Network of the agent that set the record
    pub config: Config,
}

impl SavedAgent {
    pub fn new(agent: &Agent, generation: usize, best_score: usize, best_score_seed: Option<u64>, record_agent: Option<&Agent>, config: &Config) -> Self {
        Self {
            network: agent.network.clone(),
            fitness: agent.fitness,
            generation,
            best_score,
            best_score_seed,
            record_network: record_agent.map(|a| a.network.clone()),
            config: config.clone(),
        }
    }

    /// Get the agent that set the record, if available
    pub fn record_agent(&self) -> Option<Agent> {
        self.record_network.as_ref().map(|network| Agent {
            network: network.clone(),
            fitness: 0.0,
        })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let saved: SavedAgent = serde_json::from_str(&contents)?;
        Ok(saved)
    }

    pub fn to_agent(&self) -> Agent {
        Agent {
            network: self.network.clone(),
            fitness: self.fitness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn default_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_fitness_calculation() {
        // Score 0, 10 steps, best_score=0: falls through to else case (0.2 multiplier)
        let fitness = calculate_fitness(0, 10, 0);
        assert!(fitness > 0.0);

        // Score 1 beats best_score=0: gets record bonus (base * 3 + 500)
        let fitness = calculate_fitness(1, 20, 0);
        assert!(fitness > 500.0, "Expected record bonus, got {}", fitness);

        // Score 1 ties best_score=1: gets tie bonus (base * 1.5)
        let fitness_tie = calculate_fitness(1, 20, 1);
        let base = (2.0_f64).powi(2) * 10.0 + 2.0; // 42
        assert!((fitness_tie - base * 1.5).abs() < 1.0);

        // Score 0 with best_score=1: heavy penalty (base * 0.1)
        let fitness_penalty = calculate_fitness(0, 10, 1);
        assert!(fitness_penalty < 5.0, "Expected penalty, got {}", fitness_penalty);
    }

    #[test]
    fn test_agent_plays_game() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let agent = Agent::new(&config.network, &mut rng);

        let result = agent.play_game(&config, 42, 0);

        assert!(result.fitness > 0.0);
        assert!(result.steps > 0);
    }

    #[test]
    fn test_population_creation() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let population = Population::new(config.clone(), &mut rng);

        assert_eq!(population.agents.len(), config.evolution.population_size);
    }

    #[test]
    fn test_tournament_returns_valid_agent() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let mut population = Population::new(config, &mut rng);

        // Set some fitness values
        for (i, agent) in population.agents.iter_mut().enumerate() {
            agent.fitness = i as f64;
        }

        let selected = population.tournament_select(&mut rng);
        assert!(selected.fitness >= 0.0);
    }

    #[test]
    fn test_elitism_preserves_top_n() {
        let mut config = default_config();
        config.evolution.population_size = 50;
        config.evolution.elitism_count = 5;

        let mut rng = StdRng::seed_from_u64(42);
        let mut population = Population::new(config, &mut rng);

        // Set fitness values and evaluate
        for (i, agent) in population.agents.iter_mut().enumerate() {
            agent.fitness = i as f64;
        }
        population
            .agents
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let top_fitness: Vec<f64> = population
            .agents
            .iter()
            .take(5)
            .map(|a| a.fitness)
            .collect();

        population.evolve(&mut rng);

        // Check that top agents are preserved
        let new_top: Vec<f64> = population
            .agents
            .iter()
            .take(5)
            .filter(|a| top_fitness.contains(&a.fitness))
            .map(|a| a.fitness)
            .collect();

        assert_eq!(new_top.len(), 5);
    }

    #[test]
    fn test_population_size_constant() {
        let mut config = default_config();
        config.evolution.population_size = 50;

        let mut rng = StdRng::seed_from_u64(42);
        let mut population = Population::new(config.clone(), &mut rng);

        population.evaluate(42);
        population.evolve(&mut rng);

        assert_eq!(population.agents.len(), config.evolution.population_size);
    }

    #[test]
    fn test_random_agents_complete_games() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..100 {
            let agent = Agent::new(&config.network, &mut rng);
            let result = agent.play_game(&config, i, 0);

            assert!(result.fitness > 0.0, "Game {} had zero fitness", i);
            assert!(result.steps > 0, "Game {} had zero steps", i);
        }
    }

    #[test]
    fn test_saved_agent_roundtrip() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let agent = Agent::new(&config.network, &mut rng);

        let saved = SavedAgent::new(&agent, 10, 5, Some(12345), Some(&agent), &config);
        let json = serde_json::to_string(&saved).unwrap();
        let restored: SavedAgent = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.generation, 10);
        assert_eq!(restored.best_score, 5);
        assert_eq!(restored.best_score_seed, Some(12345));
        assert_eq!(
            agent.network.get_weights_flat().len(),
            restored.network.get_weights_flat().len()
        );
    }
}
