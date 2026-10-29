# Race-time driver: only sees the car's sensors (ray distances) and speed.
import os

from claude_ai.network import Network, apply

BRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.json")
_brain = None


def drive(car, track):
    global _brain
    if _brain is None:
        _brain = Network.load(BRAIN_PATH)
    throttle, steer = _brain.decide(car.get_distances(track), car.speed)
    apply(car, throttle, steer)
