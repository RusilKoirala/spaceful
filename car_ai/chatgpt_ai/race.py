"""Race entry point; uses only the seven rays and current speed."""

from functools import lru_cache

from .brain import BRAIN_PATH, Brain, control


@lru_cache(maxsize=1)
def _policy():
    try:
        return Brain.load()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"No trained brain at {BRAIN_PATH}. Run "
            "`uv run python -m chatgpt_ai.train` before racing. "
            "Training was not run during code creation."
        ) from error


def drive(car, track):
    # Normalization uses the game settings exported by training with the brain,
    # so this module never imports the game or opens another window.
    brain, ray_length, max_speed = _race_policy()
    control(brain, car, track, ray_length, max_speed)


@lru_cache(maxsize=1)
def _race_policy():
    import json
    import math

    brain = _policy()
    with BRAIN_PATH.open() as handle:
        settings = json.load(handle)["normalization"]
    ray_length = float(settings["ray_length"])
    max_speed = float(settings["max_speed"])
    if not all(math.isfinite(value) and value > 0 for value in (ray_length, max_speed)):
        raise ValueError("Invalid saved input normalization")
    return brain, ray_length, max_speed
