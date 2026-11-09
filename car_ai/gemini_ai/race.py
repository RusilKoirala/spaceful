import os

try:
    from game import RAY_LENGTH, MAX_SPEED
except Exception:
    RAY_LENGTH = 200
    MAX_SPEED = 2

try:
    from .model import NeuralNetwork, apply_action
except ImportError:
    from gemini_ai.model import NeuralNetwork, apply_action

_BRAIN = None


def _get_brain():
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = NeuralNetwork()
        brain_path = os.path.join(os.path.dirname(__file__), "brain.json")
        if os.path.exists(brain_path):
            try:
                _BRAIN.load(brain_path)
            except Exception:
                pass
    return _BRAIN


def drive(car, track):
    """
    Called once per frame for the car during the race.
    Only accesses car.get_distances(track) and car.speed as permitted by the rules.
    Outputs the actions determined strictly by the neural network.
    """
    brain = _get_brain()

    # Exact 8 raw inputs in order: 7 ray distances / RAY_LENGTH, then speed / MAX_SPEED
    distances = car.get_distances(track)
    inputs = [d / RAY_LENGTH for d in distances] + [car.speed / MAX_SPEED]

    # Predict actions from network outputs via argmax selection
    outputs = brain.forward(inputs)
    apply_action(car, outputs)

