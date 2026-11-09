import json
import math
import os
import random


class NeuralNetwork:
    """A generic fully-connected feed-forward neural network from scratch."""

    def __init__(self, layer_sizes=None, init_sigma=0.5):
        if layer_sizes is None:
            # 8 inputs: 7 ray distances / RAY_LENGTH, speed / MAX_SPEED
            # Hidden 1: 16 neurons (tanh)
            # Hidden 2: 12 neurons (tanh)
            # Output: 6 neurons (3 for throttle, 3 for steering)
            layer_sizes = [8, 16, 12, 6]
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize with a single Gaussian distribution across all weights
        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layer_w = [
                [random.gauss(0.0, init_sigma) for _ in range(in_size)]
                for _ in range(out_size)
            ]
            layer_b = [random.gauss(0.0, init_sigma) for _ in range(out_size)]
            self.weights.append(layer_w)
            self.biases.append(layer_b)

    def forward(self, inputs):
        """Standard feed-forward pass with tanh activations on hidden layers."""
        curr = inputs
        num_layers = len(self.weights)

        for l_idx in range(num_layers):
            next_curr = []
            layer_w = self.weights[l_idx]
            layer_b = self.biases[l_idx]
            is_output_layer = l_idx == num_layers - 1

            for neuron_idx in range(len(layer_b)):
                val = layer_b[neuron_idx]
                w_row = layer_w[neuron_idx]
                for in_idx in range(len(curr)):
                    val += curr[in_idx] * w_row[in_idx]

                if not is_output_layer:
                    val = math.tanh(val)
                next_curr.append(val)

            curr = next_curr

        return curr

    def clone(self):
        """Creates an exact deep copy of this network."""
        child = NeuralNetwork(layer_sizes=self.layer_sizes)
        child.weights = [
            [[w for w in row] for row in layer] for layer in self.weights
        ]
        child.biases = [[b for b in layer] for layer in self.biases]
        return child

    def mutate(self, rate=0.15, mutation_strength=0.1):
        """Mutate weights and biases with a uniform mutation strength."""
        for layer in self.weights:
            for row in layer:
                for i in range(len(row)):
                    if random.random() < rate:
                        row[i] += random.gauss(0.0, mutation_strength)

        for layer in self.biases:
            for i in range(len(layer)):
                if random.random() < rate:
                    layer[i] += random.gauss(0.0, mutation_strength)

    @classmethod
    def crossover(cls, parent_a, parent_b):
        """Uniform crossover between two networks."""
        child = parent_a.clone()
        for l_idx in range(len(child.weights)):
            for r_idx in range(len(child.weights[l_idx])):
                for c_idx in range(len(child.weights[l_idx][r_idx])):
                    if random.random() < 0.5:
                        child.weights[l_idx][r_idx][c_idx] = (
                            parent_b.weights[l_idx][r_idx][c_idx]
                        )

            for b_idx in range(len(child.biases[l_idx])):
                if random.random() < 0.5:
                    child.biases[l_idx][b_idx] = parent_b.biases[l_idx][b_idx]

        return child

    def save(self, filepath):
        """Save network architecture and weights to a JSON file."""
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        """Load network architecture and weights from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.layer_sizes = data["layer_sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]


def apply_action(car, outputs):
    """
    Applies driving action picked solely from network outputs via argmax.
    No if rules on sensor values, no hard-coded thresholds or heuristics.
    """
    # Longitudinal choice from outputs[0:3]
    # 0: move_forward, 1: reduce_speed, 2: move_backward
    throttle_choice = max(range(3), key=lambda i: outputs[i])
    if throttle_choice == 0:
        car.move_forward()
    elif throttle_choice == 1:
        car.reduce_speed()
    elif throttle_choice == 2:
        car.move_backward()

    # Lateral choice from outputs[3:6]
    # 0: rotate left, 1: straight (no rotation), 2: rotate right
    steer_choice = max(range(3), key=lambda i: outputs[3 + i])
    if steer_choice == 0:
        car.rotate(left=True)
    elif steer_choice == 2:
        car.rotate(right=True)

