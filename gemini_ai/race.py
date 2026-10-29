"""Gemini AI Racer driver.

Provides the drive(car, track) entrypoint used by the race simulator.
Loads the trained neural network brain and controls the car using only allowed sensors.
"""

import os
from .model import NeuralNetwork, extract_features, interpret_outputs, apply_actions

# Cached trained brain instance
_BRAIN = None
_DEFAULT_TRACK = None


def get_brain():
    """Retrieves or loads the trained neural network brain singleton."""
    global _BRAIN
    if _BRAIN is not None:
        return _BRAIN

    brain_path = os.path.join(os.path.dirname(__file__), "brain.json")
    if os.path.exists(brain_path):
        try:
            _BRAIN = NeuralNetwork.load(brain_path)
            return _BRAIN
        except Exception as e:
            print(f"[Gemini AI] Warning: failed to load {brain_path} ({e}), initializing default brain.")

    # Fallback to fresh brain if file not yet trained
    _BRAIN = NeuralNetwork([11, 16, 12, 2])
    return _BRAIN


def set_brain(brain):
    """Allows programmatic setting of the active brain (useful during evaluation)."""
    global _BRAIN
    _BRAIN = brain


def _get_fallback_track():
    """Lazily loads the game track if caller invokes drive(car) without passing track."""
    global _DEFAULT_TRACK
    if _DEFAULT_TRACK is None:
        try:
            from game import load_track
            _DEFAULT_TRACK = load_track()
        except Exception:
            _DEFAULT_TRACK = None
    return _DEFAULT_TRACK


def drive(car, track=None):
    """Controls the car for one frame during the race.

    COMPLIANCE RULES:
    - Reads ONLY car.get_distances(track) and car.speed.
    - Calls AT MOST ONE of move_forward(), move_backward(), reduce_speed().
    - Calls AT MOST ONE of rotate(left=True), rotate(right=True).
    - Never sets car attributes directly.

    Args:
        car: SpaceCar instance.
        track: Track instance (optional, automatically loaded if None).
    """
    if track is None:
        track = _get_fallback_track()
        if track is None:
            return

    brain = get_brain()

    # 1. Read only permissible sensors: car.get_distances(track) and car.speed
    features = extract_features(car, track)

    # 2. Forward pass through neural network
    outputs = brain.forward(features)

    # 3. Decode outputs into allowed actions
    throttle_action, steer_action = interpret_outputs(outputs)

    # 4. Apply actions to car (at most 1 speed action, at most 1 steer action)
    apply_actions(car, throttle_action, steer_action)
