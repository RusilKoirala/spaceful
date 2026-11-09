"""Explicit genetic training: uv run python -m chatgpt_ai.train.

No training happens on import. All initial weights, including biases, use the
same Gaussian distribution. All mutations share one generation-wide strength.
Every candidate is evaluated on every starting grid slot.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics

from .brain import BRAIN_PATH, LAYERS, WEIGHT_COUNT, Brain, control


def evaluate(game, track, image, weights, render):
    brain = Brain(weights)
    scores = []
    results = []
    for slot, (x, y, angle) in enumerate(game.START_GRID):
        car = game.SpaceCar(x, y, image, "ChatGPT", angle)
        for frame in range(game.ROUND_FRAMES):
            if frame % 32 == 0:
                if any(event.type == game.pygame.QUIT for event in game.pygame.event.get()):
                    raise KeyboardInterrupt
            if not car.alive or car.finished:
                break
            control(brain, car, track, game.RAY_LENGTH, game.MAX_SPEED)
            car.step(track)
            if render:
                track.draw(game.screen)
                car.draw(game.screen, track, show_rays=True)
                game.pygame.display.flip()
                game.clock.tick(game.FPS)
        if car.finished:
            score = 3.0 + (game.ROUND_FRAMES - car.finish_frame) / game.ROUND_FRAMES
        else:
            score = max(0.0, min(1.0, car.progress / track.lap_length))
        scores.append(score)
        results.append({
            "slot": slot,
            "finished": bool(car.finished),
            "finish_frame": car.finish_frame if car.finished else None,
            "progress": float(car.progress),
        })
    # Reward reliability from arbitrary slots before optimizing lap times.
    return 0.8 * min(scores) + 0.2 * statistics.fmean(scores), results


def save_brain(weights, fitness, results, generation, seed, game):
    payload = {
        "source": "chatgpt_ai.train",
        "layers": list(LAYERS),
        "weights": list(weights),
        "normalization": {"ray_length": game.RAY_LENGTH, "max_speed": game.MAX_SPEED},
        "fitness": fitness,
        "results": results,
        "generation": generation,
        "seed": seed,
    }
    temporary = BRAIN_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n")
    temporary.replace(BRAIN_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 8:
        parser.error("Use at least one generation and a population of at least eight")
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be finite and positive")
    if not 0 < args.mutation_rate <= 1:
        parser.error("--mutation-rate must be in (0, 1]")

    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    # The documented game API is imported only when the user runs training.
    import game

    rng = random.Random(args.seed)
    population = [
        [rng.gauss(0.0, 0.5) for _ in range(WEIGHT_COUNT)]
        for _ in range(args.population)
    ]
    if args.resume:
        population[0] = list(Brain.load().weights)
    track = game.load_track()
    image = game.create_car_surface((40, 200, 150))
    best_fitness = -math.inf
    stagnant = 0
    elite_count = max(2, args.population // 12)

    try:
        for generation in range(args.generations):
            ranked = []
            for index, weights in enumerate(population):
                fitness, results = evaluate(game, track, image, weights, args.render)
                ranked.append((fitness, weights, results))
                if fitness > best_fitness:
                    best_fitness = fitness
                    stagnant = 0
                    save_brain(weights, fitness, results, generation, args.seed, game)
                print(
                    f"Generation {generation + 1}/{args.generations} "
                    f"candidate {index + 1}/{args.population}: "
                    f"fitness={fitness:.5f} best={best_fitness:.5f}",
                    flush=True,
                )
            ranked.sort(key=lambda item: item[0], reverse=True)
            stagnant += 1
            # Adapt one scalar shared by every weight, never per-weight ranges.
            strength = args.sigma * (1.0 + min(stagnant, 40) / 10.0)
            next_population = [list(item[1]) for item in ranked[:elite_count]]

            def parent():
                return max(rng.sample(ranked, 4), key=lambda item: item[0])[1]

            while len(next_population) < args.population:
                if rng.random() < 0.05:
                    child = [rng.gauss(0.0, 0.5) for _ in range(WEIGHT_COUNT)]
                else:
                    first, second = parent(), parent()
                    # Some children mutate a single parent to preserve useful
                    # hidden units; others use generic uniform crossover.
                    crossover = rng.random() < 0.25
                    child = []
                    for a, b in zip(first, second):
                        value = b if crossover and rng.random() < 0.5 else a
                        if rng.random() < args.mutation_rate:
                            value += rng.gauss(0.0, strength)
                        child.append(value)
                next_population.append(child)
            population = next_population
    except KeyboardInterrupt:
        print("Stopped. The best evaluated brain remains saved.", flush=True)
    finally:
        game.pygame.quit()


if __name__ == "__main__":
    main()
