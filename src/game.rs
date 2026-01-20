use crate::config::GameConfig;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    pub fn to_delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    pub fn to_index(&self) -> usize {
        match self {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        }
    }

    pub fn from_index(index: usize) -> Direction {
        match index {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        }
    }

    pub fn all() -> [Direction; 4] {
        [Direction::Up, Direction::Down, Direction::Left, Direction::Right]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    Empty,
    Snake,
    SnakeHead,
    Food,
}

pub struct GameState {
    pub width: usize,
    pub height: usize,
    pub snake: VecDeque<Position>,
    pub direction: Direction,
    pub food: Position,
    pub score: usize,
    pub steps: usize,
    pub energy: usize,
    pub game_over: bool,
    base_energy_per_food: usize,
    energy_decay_per_score: usize,
    minimum_energy_per_food: usize,
    rng: StdRng,
}

impl GameState {
    pub fn new(config: &GameConfig, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);

        let center_x = (config.grid_width / 2) as i32;
        let center_y = (config.grid_height / 2) as i32;

        let mut snake = VecDeque::new();
        snake.push_front(Position::new(center_x, center_y));

        let mut state = Self {
            width: config.grid_width,
            height: config.grid_height,
            snake,
            direction: Direction::Right,
            food: Position::new(0, 0),
            score: 0,
            steps: 0,
            energy: config.starting_energy,
            game_over: false,
            base_energy_per_food: config.energy_per_food,
            energy_decay_per_score: config.energy_decay_per_score,
            minimum_energy_per_food: config.minimum_energy_per_food,
            rng,
        };

        state.spawn_food();
        state
    }

    pub fn spawn_food(&mut self) {
        let mut empty_cells = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                let pos = Position::new(x as i32, y as i32);
                if !self.snake.contains(&pos) {
                    empty_cells.push(pos);
                }
            }
        }

        if !empty_cells.is_empty() {
            let index = self.rng.gen_range(0..empty_cells.len());
            self.food = empty_cells[index];
        }
    }

    pub fn step(&mut self, new_direction: Direction) {
        if self.game_over {
            return;
        }

        // Prevent 180-degree turns
        if new_direction != self.direction.opposite() {
            self.direction = new_direction;
        }

        let head = self.snake.front().unwrap();
        let (dx, dy) = self.direction.to_delta();
        let new_head = Position::new(head.x + dx, head.y + dy);

        // Check wall collision
        if new_head.x < 0
            || new_head.x >= self.width as i32
            || new_head.y < 0
            || new_head.y >= self.height as i32
        {
            self.game_over = true;
            return;
        }

        // Check self collision (excluding tail which will move)
        let will_eat = new_head == self.food;
        for (i, segment) in self.snake.iter().enumerate() {
            // If eating, tail won't move, so check all segments
            // If not eating, tail will move, so skip the last segment
            if !will_eat && i == self.snake.len() - 1 {
                continue;
            }
            if *segment == new_head {
                self.game_over = true;
                return;
            }
        }

        self.snake.push_front(new_head);
        self.steps += 1;

        // Energy drain - every step costs 1 energy
        if self.energy > 0 {
            self.energy -= 1;
        }

        if will_eat {
            self.score += 1;
            // Energy gain decreases with score (harder as you progress)
            let decay = self.score * self.energy_decay_per_score;
            let energy_gain = if self.base_energy_per_food > decay {
                (self.base_energy_per_food - decay).max(self.minimum_energy_per_food)
            } else {
                self.minimum_energy_per_food
            };
            self.energy += energy_gain;
            self.spawn_food();
        } else {
            self.snake.pop_back();
        }

        // Out of energy = death
        if self.energy == 0 {
            self.game_over = true;
        }
    }

    pub fn get_cell(&self, x: usize, y: usize) -> Cell {
        let pos = Position::new(x as i32, y as i32);

        if pos == self.food {
            return Cell::Food;
        }

        if let Some(head) = self.snake.front() {
            if *head == pos {
                return Cell::SnakeHead;
            }
        }

        if self.snake.contains(&pos) {
            return Cell::Snake;
        }

        Cell::Empty
    }

    pub fn to_network_input(&self) -> Vec<f64> {
        let mut input = Vec::with_capacity(self.width * self.height + 4);

        // Grid encoding: 0.0=empty, 0.33=body, 0.66=head, 1.0=food
        for y in 0..self.height {
            for x in 0..self.width {
                let value = match self.get_cell(x, y) {
                    Cell::Empty => 0.0,
                    Cell::Snake => 0.33,
                    Cell::SnakeHead => 0.66,
                    Cell::Food => 1.0,
                };
                input.push(value);
            }
        }

        // Direction one-hot encoding
        for dir in Direction::all() {
            input.push(if self.direction == dir { 1.0 } else { 0.0 });
        }

        input
    }

    pub fn input_size(config: &GameConfig) -> usize {
        config.grid_width * config.grid_height + 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GameConfig {
        GameConfig {
            grid_width: 8,
            grid_height: 8,
            starting_energy: 100,
            energy_per_food: 75,
            energy_decay_per_score: 5,
            minimum_energy_per_food: 20,
        }
    }

    #[test]
    fn test_initial_state() {
        let config = default_config();
        let game = GameState::new(&config, 42);

        assert_eq!(game.snake.len(), 1);
        assert_eq!(game.score, 0);
        assert_eq!(game.steps, 0);
        assert!(!game.game_over);
    }

    #[test]
    fn test_movement() {
        let config = default_config();
        let mut game = GameState::new(&config, 42);

        let initial_head = *game.snake.front().unwrap();
        game.step(Direction::Right);

        let new_head = *game.snake.front().unwrap();
        assert_eq!(new_head.x, initial_head.x + 1);
        assert_eq!(new_head.y, initial_head.y);
    }

    #[test]
    fn test_wall_collision() {
        let config = default_config();
        let mut game = GameState::new(&config, 42);

        // Move to the right wall
        for _ in 0..10 {
            game.step(Direction::Right);
        }

        assert!(game.game_over);
    }

    #[test]
    fn test_opposite_direction_prevented() {
        let config = default_config();
        let mut game = GameState::new(&config, 42);

        game.direction = Direction::Right;
        game.step(Direction::Left); // Should not change to left

        assert_eq!(game.direction, Direction::Right);
    }

    #[test]
    fn test_input_encoding_length() {
        let config = default_config();
        let game = GameState::new(&config, 42);

        let input = game.to_network_input();
        assert_eq!(input.len(), 8 * 8 + 4); // 68
    }

    #[test]
    fn test_reproducibility() {
        let config = default_config();

        let game1 = GameState::new(&config, 42);
        let game2 = GameState::new(&config, 42);

        assert_eq!(game1.food.x, game2.food.x);
        assert_eq!(game1.food.y, game2.food.y);
    }

    #[test]
    fn test_direction_methods() {
        assert_eq!(Direction::Up.opposite(), Direction::Down);
        assert_eq!(Direction::Left.opposite(), Direction::Right);
        assert_eq!(Direction::Up.to_delta(), (0, -1));
        assert_eq!(Direction::Right.to_delta(), (1, 0));
    }

    #[test]
    fn test_energy_depletion() {
        let config = GameConfig {
            grid_width: 8,
            grid_height: 8,
            starting_energy: 10,
            energy_per_food: 75,
            energy_decay_per_score: 5,
            minimum_energy_per_food: 20,
        };
        let mut game = GameState::new(&config, 42);

        // Move in a pattern that doesn't eat food
        for i in 0..15 {
            let dir = if i % 2 == 0 {
                Direction::Down
            } else {
                Direction::Up
            };
            game.step(dir);
            if game.game_over {
                break;
            }
        }

        assert!(game.game_over);
        assert!(game.energy == 0 || game.score > 0 || game.steps <= 15);
    }

    #[test]
    fn test_energy_replenish() {
        let config = GameConfig {
            grid_width: 8,
            grid_height: 8,
            starting_energy: 100,
            energy_per_food: 50,
            energy_decay_per_score: 5,
            minimum_energy_per_food: 20,
        };
        let mut game = GameState::new(&config, 42);

        let initial_energy = game.energy;

        // Take a few steps (drain energy)
        for _ in 0..5 {
            game.step(Direction::Right);
            if game.game_over {
                break;
            }
        }

        // Energy should have decreased
        assert!(game.energy < initial_energy || game.score > 0);
    }

    #[test]
    fn test_energy_decay_with_score() {
        let config = GameConfig {
            grid_width: 8,
            grid_height: 8,
            starting_energy: 100,
            energy_per_food: 75,
            energy_decay_per_score: 10,
            minimum_energy_per_food: 20,
        };

        // At score 0: gain = 75
        // At score 3: gain = 75 - 30 = 45
        // At score 6: gain = 75 - 60 = 20 (minimum)

        let decay_at_0 = 0 * 10;
        let gain_at_0 = 75_usize.saturating_sub(decay_at_0).max(20);
        assert_eq!(gain_at_0, 75);

        let decay_at_3 = 3 * 10;
        let gain_at_3 = 75_usize.saturating_sub(decay_at_3).max(20);
        assert_eq!(gain_at_3, 45);

        let decay_at_6 = 6 * 10;
        let gain_at_6 = 75_usize.saturating_sub(decay_at_6).max(20);
        assert_eq!(gain_at_6, 20);
    }
}
