"""Sensor-only neural driver. All learned parameters live beside this file."""

import json
import math
from pathlib import Path


class Brain:
    """A small, symmetric neural network implemented with plain Python.

    Three shared tanh neurons process the opposing sensor pairs. A fourth
    neuron estimates corner urgency from the forward ray and speed. The
    steering and throttle heads operate only on those activations.
    """

    def __init__(self, weights):
        self.weights = list(weights)

    def forward(self, distances, speed):
        w = self.weights
        rays = [max(0.0, min(1.0, d / 200.0)) for d in distances]
        side = math.tanh(w[0] * (rays[0] - rays[6]))
        diagonal = math.tanh(w[1] * (rays[1] - rays[5]))
        ahead = math.tanh(w[2] * (rays[2] - rays[4]))
        urgency = math.tanh(w[3] * (1.0 - rays[3]) + w[4] * speed / 2.0 + w[5])
        steering = math.tanh(w[6] * side + w[7] * diagonal + w[8] * ahead)
        throttle = math.tanh(w[9] + w[10] * urgency + w[11] * abs(steering))
        return steering, throttle

    def drive(self, car, track):
        steering, throttle = self.forward(car.get_distances(track), car.speed)
        if throttle >= 0.0:
            car.move_forward()
        else:
            car.reduce_speed()
        if steering > self.weights[12]:
            car.rotate(left=True)
        elif steering < -self.weights[12]:
            car.rotate(right=True)


_brain = None


def drive(car, track):
    """Called once per frame; no race-time access to position or progress."""
    global _brain
    if _brain is None:
        with Path(__file__).with_name("brain.json").open() as source:
            _brain = Brain(json.load(source)["weights"])
    _brain.drive(car, track)
