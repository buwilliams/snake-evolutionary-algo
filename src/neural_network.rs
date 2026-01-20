use crate::config::{ActivationType, NetworkConfig};
use crate::game::Direction;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>, // [output][input]
    pub biases: Vec<f64>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, rng: &mut StdRng) -> Self {
        // Xavier/He initialization
        let std_dev = (2.0 / input_size as f64).sqrt();

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| gaussian(rng, std_dev))
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(w, b)| {
                w.iter().zip(input).map(|(wi, xi)| wi * xi).sum::<f64>() + b
            })
            .collect()
    }

    pub fn input_size(&self) -> usize {
        if self.weights.is_empty() {
            0
        } else {
            self.weights[0].len()
        }
    }

    pub fn output_size(&self) -> usize {
        self.weights.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub activation: ActivationType,
}

impl NeuralNetwork {
    pub fn new(config: &NetworkConfig, rng: &mut StdRng) -> Self {
        Self::with_hidden_layers(config.input_size, &config.hidden_layers, config.output_size, config.activation, rng)
    }

    /// Create a network with specific hidden layer sizes
    pub fn with_hidden_layers(
        input_size: usize,
        hidden_layers: &[usize],
        output_size: usize,
        activation: ActivationType,
        rng: &mut StdRng,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        // Hidden layers
        for &hidden_size in hidden_layers {
            layers.push(Layer::new(prev_size, hidden_size, rng));
            prev_size = hidden_size;
        }

        // Output layer
        layers.push(Layer::new(prev_size, output_size, rng));

        Self { layers, activation }
    }

    /// Get the hidden layer sizes (excludes output layer)
    pub fn hidden_layer_sizes(&self) -> Vec<usize> {
        self.layers.iter()
            .take(self.layers.len().saturating_sub(1))
            .map(|l| l.output_size())
            .collect()
    }

    /// Grow the network by adding neurons to an existing layer or adding a new layer
    /// Returns true if growth occurred
    pub fn grow(&mut self, rng: &mut StdRng) -> bool {
        if self.layers.len() < 2 {
            return false; // Need at least input->output
        }

        let hidden_count = self.layers.len() - 1;

        // Decide: add neurons to existing layer (70%) or add new layer (30%)
        if hidden_count > 0 && rng.gen::<f64>() < 0.7 {
            // Add neurons to a random hidden layer
            let layer_idx = rng.gen_range(0..hidden_count);
            self.add_neurons_to_layer(layer_idx, 8, rng); // Add 8 neurons
            true
        } else if hidden_count < 4 {
            // Add a new hidden layer (max 4 hidden layers)
            self.add_hidden_layer(16, rng); // New layer with 16 neurons
            true
        } else {
            // Already at max layers, add neurons instead
            let layer_idx = rng.gen_range(0..hidden_count);
            self.add_neurons_to_layer(layer_idx, 8, rng);
            true
        }
    }

    /// Add neurons to an existing hidden layer
    fn add_neurons_to_layer(&mut self, layer_idx: usize, count: usize, rng: &mut StdRng) {
        if layer_idx >= self.layers.len() - 1 {
            return; // Don't modify output layer
        }

        let layer = &mut self.layers[layer_idx];
        let input_size = layer.input_size();
        let std_dev = (2.0 / input_size as f64).sqrt();

        // Add new neurons (new rows in weights matrix, new biases)
        for _ in 0..count {
            let new_weights: Vec<f64> = (0..input_size)
                .map(|_| gaussian(rng, std_dev) * 0.1) // Small initial weights
                .collect();
            layer.weights.push(new_weights);
            layer.biases.push(0.0);
        }

        // Update next layer to accept more inputs
        if layer_idx + 1 < self.layers.len() {
            let next_layer = &mut self.layers[layer_idx + 1];
            let next_std_dev = (2.0 / next_layer.input_size() as f64).sqrt();

            for row in &mut next_layer.weights {
                for _ in 0..count {
                    row.push(gaussian(rng, next_std_dev) * 0.1); // Small initial weights
                }
            }
        }
    }

    /// Add a new hidden layer before the output layer
    fn add_hidden_layer(&mut self, size: usize, rng: &mut StdRng) {
        if self.layers.is_empty() {
            return;
        }

        let output_layer_idx = self.layers.len() - 1;
        let prev_size = if output_layer_idx > 0 {
            self.layers[output_layer_idx - 1].output_size()
        } else {
            self.layers[0].input_size()
        };

        // Create new hidden layer
        let new_layer = Layer::new(prev_size, size, rng);

        // Create new output layer that takes input from new hidden layer
        let output_size = self.layers[output_layer_idx].output_size();
        let new_output_layer = Layer::new(size, output_size, rng);

        // Replace output layer with: new_hidden -> new_output
        self.layers.pop(); // Remove old output
        self.layers.push(new_layer);
        self.layers.push(new_output_layer);
    }

    fn activate(&self, x: f64) -> f64 {
        match self.activation {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Tanh => x.tanh(),
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(&current);

            // Apply activation to all layers except the last (output layer)
            if i < self.layers.len() - 1 {
                current = output.into_iter().map(|x| self.activate(x)).collect();
            } else {
                current = output; // Raw logits for output layer
            }
        }

        current
    }

    pub fn decide(&self, input: &[f64]) -> Direction {
        let output = self.forward(input);

        // Argmax
        let (max_idx, _) = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Direction::from_index(max_idx)
    }

    pub fn get_weights_flat(&self) -> Vec<f64> {
        let mut weights = Vec::new();

        for layer in &self.layers {
            for row in &layer.weights {
                weights.extend(row);
            }
            weights.extend(&layer.biases);
        }

        weights
    }

    pub fn set_weights_flat(&mut self, weights: &[f64]) {
        let mut idx = 0;

        for layer in &mut self.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    *w = weights[idx];
                    idx += 1;
                }
            }
            for b in &mut layer.biases {
                *b = weights[idx];
                idx += 1;
            }
        }
    }

    pub fn weight_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.iter().map(|r| r.len()).sum::<usize>() + l.biases.len())
            .sum()
    }

    pub fn mutate(&mut self, rate: f64, strength: f64, rng: &mut StdRng) {
        for layer in &mut self.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    if rng.gen::<f64>() < rate {
                        *w += gaussian(rng, strength);
                    }
                }
            }
            for b in &mut layer.biases {
                if rng.gen::<f64>() < rate {
                    *b += gaussian(rng, strength);
                }
            }
        }
    }

    /// Crossover two networks - handles different architectures
    pub fn crossover(&self, other: &NeuralNetwork, rng: &mut StdRng) -> NeuralNetwork {
        let self_sizes = self.hidden_layer_sizes();
        let other_sizes = other.hidden_layer_sizes();

        // If same architecture, do standard crossover
        if self_sizes == other_sizes {
            let mut child = self.clone();
            let self_weights = self.get_weights_flat();
            let other_weights = other.get_weights_flat();

            let child_weights: Vec<f64> = self_weights
                .iter()
                .zip(&other_weights)
                .map(|(a, b)| if rng.gen::<bool>() { *a } else { *b })
                .collect();

            child.set_weights_flat(&child_weights);
            return child;
        }

        // Different architectures: child inherits from one parent randomly
        // with slight preference for the fitter (assumed to be self)
        if rng.gen::<f64>() < 0.6 {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Get total number of neurons in hidden layers (complexity measure)
    pub fn complexity(&self) -> usize {
        self.hidden_layer_sizes().iter().sum()
    }
}

/// Box-Muller transform for generating Gaussian random numbers
fn gaussian(rng: &mut StdRng, std_dev: f64) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();

    // Avoid log(0)
    let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * std_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn default_config() -> NetworkConfig {
        NetworkConfig {
            input_size: 68,
            hidden_layers: vec![64, 32],
            output_size: 4,
            activation: ActivationType::ReLU,
        }
    }

    #[test]
    fn test_layer_dimensions() {
        let mut rng = StdRng::seed_from_u64(42);
        let layer = Layer::new(68, 64, &mut rng);

        assert_eq!(layer.input_size(), 68);
        assert_eq!(layer.output_size(), 64);
    }

    #[test]
    fn test_forward_pass_output_size() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let network = NeuralNetwork::new(&config, &mut rng);

        let input = vec![0.0; 68];
        let output = network.forward(&input);

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_weight_roundtrip() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let network = NeuralNetwork::new(&config, &mut rng);

        let original_weights = network.get_weights_flat();
        let mut restored = network.clone();
        restored.set_weights_flat(&original_weights);
        let restored_weights = restored.get_weights_flat();

        assert_eq!(original_weights.len(), restored_weights.len());
        for (a, b) in original_weights.iter().zip(&restored_weights) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mutation_modifies_weights() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let mut network = NeuralNetwork::new(&config, &mut rng);

        let original_weights = network.get_weights_flat();
        network.mutate(1.0, 0.3, &mut rng); // 100% mutation rate
        let mutated_weights = network.get_weights_flat();

        let mut different_count = 0;
        for (a, b) in original_weights.iter().zip(&mutated_weights) {
            if (a - b).abs() > 1e-10 {
                different_count += 1;
            }
        }

        assert!(different_count > 0);
    }

    #[test]
    fn test_decide_returns_valid_direction() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let network = NeuralNetwork::new(&config, &mut rng);

        let input = vec![0.5; 68];
        let direction = network.decide(&input);

        assert!(matches!(
            direction,
            Direction::Up | Direction::Down | Direction::Left | Direction::Right
        ));
    }

    #[test]
    fn test_crossover() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = NeuralNetwork::new(&config, &mut rng);
        let parent2 = NeuralNetwork::new(&config, &mut rng);
        let child = parent1.crossover(&parent2, &mut rng);

        let p1_weights = parent1.get_weights_flat();
        let p2_weights = parent2.get_weights_flat();
        let c_weights = child.get_weights_flat();

        // Child weights should come from either parent
        for (i, c) in c_weights.iter().enumerate() {
            let from_p1 = (c - p1_weights[i]).abs() < 1e-10;
            let from_p2 = (c - p2_weights[i]).abs() < 1e-10;
            assert!(from_p1 || from_p2);
        }
    }

    #[test]
    fn test_weight_count() {
        let config = default_config();
        let mut rng = StdRng::seed_from_u64(42);
        let network = NeuralNetwork::new(&config, &mut rng);

        // 68*64 + 64 + 64*32 + 32 + 32*4 + 4
        let expected = 68 * 64 + 64 + 64 * 32 + 32 + 32 * 4 + 4;
        assert_eq!(network.weight_count(), expected);
        assert_eq!(network.get_weights_flat().len(), expected);
    }
}
