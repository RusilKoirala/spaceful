# A small fully connected neural network, written from scratch (standard library only).
import json
import math
import random

from settings import MAX_SPEED, RAY_LENGTH

LAYERS = [8, 14, 10, 7]   # 7 rays + speed  ->  hidden  ->  4 throttle choices + 3 steering choices
THROTTLE = ("forward", "coast", "brake", "backward")
STEER = ("left", "straight", "right")


def n_weights(layers=LAYERS):
    return sum((a + 1) * b for a, b in zip(layers, layers[1:]))


class Network:
    def __init__(self, genome=None, layers=LAYERS):
        self.layers = list(layers)
        if genome is None:
            genome = [random.gauss(0, 0.5) for _ in range(n_weights(self.layers))]
        self.genome = list(genome)
        # unpack the flat genome into per-layer weight rows (last value of each row is the bias)
        self.weights = []
        i = 0
        for a, b in zip(self.layers, self.layers[1:]):
            layer = []
            for _ in range(b):
                layer.append(self.genome[i:i + a + 1])
                i += a + 1
            self.weights.append(layer)

    def forward(self, inputs):
        values = inputs
        last = len(self.weights) - 1
        for li, layer in enumerate(self.weights):
            out = []
            for row in layer:
                s = row[-1]
                for w, v in zip(row, values):
                    s += w * v
                out.append(s if li == last else math.tanh(s))
            values = out
        return values

    def decide(self, distances, speed):
        inputs = [d / RAY_LENGTH for d in distances] + [speed / MAX_SPEED]
        out = self.forward(inputs)
        throttle = max(range(4), key=lambda k: out[k])
        steer = max(range(3), key=lambda k: out[4 + k])
        return THROTTLE[throttle], STEER[steer]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"layers": self.layers, "genome": self.genome}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["genome"], data["layers"])


def apply(car, throttle, steer):
    # at most one throttle call and at most one rotate call per frame
    if throttle == "forward":
        car.move_forward()
    elif throttle == "brake":
        car.reduce_speed()
    elif throttle == "backward":
        car.move_backward()
    if steer == "left":
        car.rotate(left=True)
    elif steer == "right":
        car.rotate(right=True)
