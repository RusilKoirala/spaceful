"""Neural network implementation built from scratch for Gemini AI racer.

Standard library only (math, random, json).
No external ML frameworks.
"""

import json
import math
import random


class NeuralNetwork:
    """Multi-layer Perceptron (MLP) with tanh activations.

    Designed for neuroevolution, genetic algorithms, and continuous control.
    """

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            # 11 input features: 7 rays + 3 ray deltas + speed
            # 16 hidden layer 1
            # 12 hidden layer 2
            # 2 outputs: [steer, throttle/brake]
            layer_sizes = [11, 16, 12, 2]

        self.layer_sizes = list(layer_sizes)
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            # Xavier / Glorot uniform initialization
            limit = math.sqrt(6.0 / (n_in + n_out))
            w = [[random.uniform(-limit, limit) for _ in range(n_out)] for _ in range(n_in)]
            b = [0.0 for _ in range(n_out)]
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        """Forward pass through the network with tanh activations.

        Args:
            inputs: List of float inputs of length layer_sizes[0].

        Returns:
            List of float outputs of length layer_sizes[-1], each in [-1.0, 1.0].
        """
        current = inputs
        for l in range(len(self.weights)):
            w = self.weights[l]
            b = self.biases[l]
            n_in = len(current)
            n_out = len(b)
            next_layer = [0.0] * n_out

            for j in range(n_out):
                acc = b[j]
                for i in range(n_in):
                    acc += current[i] * w[i][j]
                next_layer[j] = math.tanh(acc)

            current = next_layer
        return current

    def clone(self):
        """Creates a deep copy of this neural network."""
        cloned = NeuralNetwork(self.layer_sizes)
        cloned.weights = [[[val for val in row] for row in layer] for layer in self.weights]
        cloned.biases = [[val for val in layer] for layer in self.biases]
        return cloned

    def mutate(self, rate=0.08, power=0.15):
        """Applies Gaussian mutation to weights and biases in-place.

        Args:
            rate: Probability of mutating each individual parameter.
            power: Standard deviation of Gaussian perturbation.
        """
        for l in range(len(self.weights)):
            w = self.weights[l]
            n_in = len(w)
            n_out = len(w[0])
            for i in range(n_in):
                for j in range(n_out):
                    if random.random() < rate:
                        if random.random() < 0.05:
                            # Occasional random reset to avoid getting trapped in local minima
                            limit = math.sqrt(6.0 / (n_in + n_out))
                            w[i][j] = random.uniform(-limit, limit)
                        else:
                            w[i][j] += random.gauss(0.0, power)
                            # Clip weights to prevent explosion
                            if w[i][j] > 4.0:
                                w[i][j] = 4.0
                            elif w[i][j] < -4.0:
                                w[i][j] = -4.0

            b = self.biases[l]
            for j in range(len(b)):
                if random.random() < rate:
                    b[j] += random.gauss(0.0, power)
                    if b[j] > 4.0:
                        b[j] = 4.0
                    elif b[j] < -4.0:
                        b[j] = -4.0

    @classmethod
    def crossover(cls, parent_a, parent_b):
        """Performs uniform crossover between two parent networks.

        Returns:
            A new child NeuralNetwork.
        """
        child = NeuralNetwork(parent_a.layer_sizes)
        for l in range(len(parent_a.weights)):
            w_a = parent_a.weights[l]
            w_b = parent_b.weights[l]
            w_c = child.weights[l]
            n_in = len(w_a)
            n_out = len(w_a[0])

            for i in range(n_in):
                for j in range(n_out):
                    w_c[i][j] = w_a[i][j] if random.random() < 0.5 else w_b[i][j]

            b_a = parent_a.biases[l]
            b_b = parent_b.biases[l]
            b_c = child.biases[l]
            for j in range(len(b_a)):
                b_c[j] = b_a[j] if random.random() < 0.5 else b_b[j]

        return child

    def to_dict(self):
        """Serializes network parameters to a JSON-compatible dictionary."""
        return {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data):
        """Constructs a network from a parameter dictionary."""
        net = cls(data["layer_sizes"])
        net.weights = data["weights"]
        net.biases = data["biases"]
        return net

    def save(self, filepath):
        """Saves network weights to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Loads network weights from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def extract_features(car, track):
    """Extracts normalized sensor features from car and track.

    STRICT COMPLIANCE:
    Reads ONLY car.get_distances(track) and car.speed.
    Never accesses car.x, car.y, car.progress, car.angle, or other prohibited state.

    Returns:
        List of 11 float features in approximately [-1.0, 1.0].
    """
    # 7 distance rays at angles [90, 60, 30, 0, -30, -60, -90]
    # Each distance is in [0, RAY_LENGTH] where RAY_LENGTH = 200
    distances = car.get_distances(track)
    ray_norm = [d / 200.0 for d in distances]

    # Differential features for direct left/right steering balance
    diff_90 = ray_norm[0] - ray_norm[6]  # full left vs full right
    diff_60 = ray_norm[1] - ray_norm[5]  # diagonal left vs diagonal right
    diff_30 = ray_norm[2] - ray_norm[4]  # forward-angle left vs forward-angle right

    # Speed normalized by MAX_SPEED (2.0)
    # Range: [-0.5, 1.0]
    speed_norm = car.speed / 2.0

    return [
        ray_norm[0],  # left 90°
        ray_norm[1],  # left 60°
        ray_norm[2],  # left 30°
        ray_norm[3],  # center 0° (ahead runway)
        ray_norm[4],  # right -30°
        ray_norm[5],  # right -60°
        ray_norm[6],  # right -90°
        diff_90,
        diff_60,
        diff_30,
        speed_norm,
    ]


def interpret_outputs(outputs):
    """Interprets neural network outputs into discrete car control actions.

    Outputs:
        outputs[0]: steer control in [-1.0, 1.0]
        outputs[1]: throttle / brake control in [-1.0, 1.0]

    Returns:
        tuple (throttle_action, steer_action):
            throttle_action: 'forward', 'backward', 'reduce', or None
            steer_action: 'left', 'right', or None
    """
    steer_val = outputs[0]
    speed_val = outputs[1]

    # Steering selection (at most one rotate call)
    if steer_val > 0.05:
        steer_action = "left"
    elif steer_val < -0.05:
        steer_action = "right"
    else:
        steer_action = None

    # Speed selection (at most one move_forward / move_backward / reduce_speed call)
    if speed_val > 0.05:
        throttle_action = "forward"
    elif speed_val < -0.45:
        throttle_action = "backward"
    elif speed_val < -0.10:
        throttle_action = "reduce"
    else:
        throttle_action = None

    return throttle_action, steer_action


def apply_actions(car, throttle_action, steer_action):
    """Executes actions on car while strictly following race rules.

    Rules:
    - At most ONE of move_forward(), move_backward(), reduce_speed()
    - At most ONE of rotate(left=True), rotate(right=True)
    - Never set car attributes directly.
    """
    # 1. Steering action
    if steer_action == "left":
        car.rotate(left=True)
    elif steer_action == "right":
        car.rotate(right=True)

    # 2. Throttle / Speed action
    if throttle_action == "forward":
        car.move_forward()
    elif throttle_action == "backward":
        car.move_backward()
    elif throttle_action == "reduce":
        car.reduce_speed()
