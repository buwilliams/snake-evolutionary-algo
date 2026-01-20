use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub max_steps_without_food: usize,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            grid_width: 8,
            grid_height: 8,
            max_steps_without_food: 100,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    ReLU,
    Tanh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub activation: ActivationType,
}

impl NetworkConfig {
    pub fn new(game_config: &GameConfig) -> Self {
        let input_size = game_config.grid_width * game_config.grid_height + 4;
        Self {
            input_size,
            hidden_layers: vec![64, 32],
            output_size: 4,
            activation: ActivationType::ReLU,
        }
    }

    pub fn validate(&self, game_config: &GameConfig) -> Result<(), String> {
        let expected_input = game_config.grid_width * game_config.grid_height + 4;
        if self.input_size != expected_input {
            return Err(format!(
                "Input size mismatch: expected {} for {}x{} grid, got {}",
                expected_input, game_config.grid_width, game_config.grid_height, self.input_size
            ));
        }
        if self.output_size != 4 {
            return Err(format!("Output size must be 4 (directions), got {}", self.output_size));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub tournament_size: usize,
    pub elitism_count: usize,
    pub mutation_rate: f64,
    pub mutation_strength: f64,
    pub crossover_rate: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 500,
            tournament_size: 5,
            elitism_count: 10,
            mutation_rate: 0.1,
            mutation_strength: 0.3,
            crossover_rate: 0.7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub generations: usize,
    pub games_per_evaluation: usize,
    pub save_interval: usize,
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            generations: 500,
            games_per_evaluation: 3,
            save_interval: 50,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub game: GameConfig,
    pub network: NetworkConfig,
    pub evolution: EvolutionConfig,
    pub training: TrainingConfig,
}

impl Default for Config {
    fn default() -> Self {
        let game = GameConfig::default();
        let network = NetworkConfig::new(&game);
        Self {
            game,
            network,
            evolution: EvolutionConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        self.network.validate(&self.game)?;

        if self.evolution.elitism_count >= self.evolution.population_size {
            return Err("Elitism count must be less than population size".to_string());
        }
        if self.evolution.tournament_size > self.evolution.population_size {
            return Err("Tournament size must not exceed population size".to_string());
        }

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.game.grid_width, restored.game.grid_width);
        assert_eq!(config.network.input_size, restored.network.input_size);
        assert_eq!(config.evolution.population_size, restored.evolution.population_size);
    }

    #[test]
    fn test_input_size_calculation() {
        let game = GameConfig::default();
        let network = NetworkConfig::new(&game);
        assert_eq!(network.input_size, 8 * 8 + 4); // 68
    }

    #[test]
    fn test_validation_catches_mismatch() {
        let game = GameConfig::default();
        let mut network = NetworkConfig::new(&game);
        network.input_size = 100; // Wrong size

        assert!(network.validate(&game).is_err());
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }
}
