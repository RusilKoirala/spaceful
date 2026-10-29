"""A fully connected neural policy implemented with the standard library."""

import json
import math
from pathlib import Path


LAYERS = (8, 16, 5)
WEIGHT_COUNT = sum((a + 1) * b for a, b in zip(LAYERS, LAYERS[1:]))
BRAIN_PATH = Path(__file__).with_name("brain.json")


class Brain:
    def __init__(self, weights):
        if len(weights) != WEIGHT_COUNT:
            raise ValueError(f"Expected {WEIGHT_COUNT} weights")
        self.weights = tuple(float(value) for value in weights)
        if not all(math.isfinite(value) for value in self.weights):
            raise ValueError("Weights must be finite")
        self.rows = []
        offset = 0
        for inputs, outputs in zip(LAYERS, LAYERS[1:]):
            rows = []
            for _ in range(outputs):
                rows.append(self.weights[offset:offset + inputs + 1])
                offset += inputs + 1
            self.rows.append(rows)

    def forward(self, inputs):
        if len(inputs) != LAYERS[0]:
            raise ValueError("Expected seven ray distances and one speed input")
        values = inputs
        for layer_index, rows in enumerate(self.rows):
            values = [
                row[-1] + sum(weight * value for weight, value in zip(row, values))
                for row in rows
            ]
            if layer_index < len(self.rows) - 1:
                values = [math.tanh(value) for value in values]
        return values

    @classmethod
    def load(cls, path=BRAIN_PATH):
        with Path(path).open() as handle:
            payload = json.load(handle)
        if payload.get("layers") != list(LAYERS):
            raise ValueError("Saved brain has an incompatible architecture")
        if payload.get("source") != "chatgpt_ai.train":
            raise ValueError("Expected a brain saved by chatgpt_ai.train")
        return cls(payload["weights"])


def control(brain, car, track, ray_length, max_speed):
    inputs = [distance / ray_length for distance in car.get_distances(track)]
    inputs.append(car.speed / max_speed)
    outputs = brain.forward(inputs)

    # Both decisions are direct argmaxes of network outputs. A steering action
    # of zero means no rotation; neither choice examines sensors or car state.
    throttle = max(range(2), key=lambda index: outputs[index])
    steering = max(range(3), key=lambda index: outputs[index + 2])
    if throttle == 0:
        car.move_forward()
    else:
        car.reduce_speed()
    if steering == 1:
        car.rotate(left=True)
    elif steering == 2:
        car.rotate(right=True)
