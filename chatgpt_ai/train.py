"""Train a neural sensor controller using standard-library evolution.

Run from the project root: uv run python -m chatgpt_ai.train
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from game import (
    ROUND_FRAMES,
    START_GRID,
    SpaceCar,
    create_car_surface,
    load_track,
)
from .race import Brain


ROOT = Path(__file__).resolve().parent
INITIAL = [2.0, 2.0, 2.0, 3.0, 1.0, -2.0, 0.2, 0.8, 1.6, 3.0, 0.0, 0.0, 0.06]


def evaluate(weights, track, slots=(0,), frame_limit=ROUND_FRAMES):
    brain = Brain(weights)
    results = []
    surface = create_car_surface((70, 210, 180))
    for slot in slots:
        x, y, angle = START_GRID[slot]
        car = SpaceCar(x, y, surface, "ChatGPT", angle)
        for _ in range(frame_limit):
            if not car.alive:
                break
            brain.drive(car, track)
            car.step(track)
            if car.finished:
                break
        results.append({
            "slot": slot,
            "finished": bool(car.finished),
            "frames": car.finish_frame if car.finished else car.frames_alive,
            "progress": round(car.progress, 3),
            "crashed": bool(car.crashed),
        })
    scores = [
        2.0 * track.lap_length + (ROUND_FRAMES - r["frames"]) * 2.0
        if r["finished"] else r["progress"]
        for r in results
    ]
    # Favor the worst starting slot, then reward overall lap speed.
    return min(scores) * 0.8 + sum(scores) / len(scores) * 0.2, results


def constrain(weights):
    w = list(weights)
    for i in (0, 1, 2):
        w[i] = max(0.1, min(12.0, w[i]))
    for i in (6, 7, 8):
        w[i] = max(-4.0, min(8.0, w[i]))
    w[12] = max(0.002, min(0.6, w[12]))
    return w


def save(weights, score, results, evaluations, elapsed):
    payload = {
        "architecture": "7 rays + speed; 4 tanh hidden neurons; steering and throttle heads",
        "weights": weights,
        "score": score,
        "validation": results,
        "evaluations": evaluations,
        "training_seconds": round(elapsed, 2),
    }
    destination = ROOT / "brain.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=45)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=87191)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    track = load_track()
    started = time.monotonic()
    weights = INITIAL[:]
    if (ROOT / "brain.json").exists():
        weights = json.loads((ROOT / "brain.json").read_text())["weights"]
    if args.validate:
        score, results = evaluate(weights, track, range(len(START_GRID)))
        print(json.dumps({"score": score, "results": results}, indent=2), flush=True)
        return

    score, results = evaluate(weights, track)
    evaluations = 1
    best = weights[:]
    elite = [(score, best)]
    print(f"Initial score {score:.1f}: {results}", flush=True)
    scales = [0.6, 0.6, 0.6, 0.5, 0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.4, 0.4, 0.035]
    for generation in range(args.generations):
        candidates = elite[:]
        temperature = max(0.12, 1.0 - generation / max(1, args.generations))
        for _ in range(args.population):
            parent = rng.choice(elite)[1]
            child = constrain([
                value + rng.gauss(0, scale * temperature)
                for value, scale in zip(parent, scales)
            ])
            fitness, _ = evaluate(child, track)
            evaluations += 1
            candidates.append((fitness, child))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elite = candidates[:4]
        score, best = elite[0]
        print(f"Generation {generation + 1}/{args.generations}: {score:.1f}", flush=True)
        # Persist a usable model throughout training, including interruption.
        _, results = evaluate(best, track)
        save(best, score, results, evaluations, time.monotonic() - started)

    score, results = evaluate(best, track, range(len(START_GRID)))
    save(best, score, results, evaluations, time.monotonic() - started)
    print(json.dumps({"score": score, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
