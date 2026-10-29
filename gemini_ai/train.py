"""Gemini AI Racer training script.

Trains the neural network from scratch using neuroevolution (genetic algorithm).
Runs headlessly with:
    uv run python -m gemini_ai.train
"""

import argparse
import os
import random
import sys
import time

# Headless training: do not open a graphical window
os.environ["SDL_VIDEODRIVER"] = "dummy"

from game import (
    START_GRID,
    ROUND_FRAMES,
    create_car_surface,
    load_track,
    SpaceCar,
)
from .model import (
    NeuralNetwork,
    extract_features,
    interpret_outputs,
    apply_actions,
)


def evaluate_car(network, track, car_image, grid_slot_idx=0, max_frames=ROUND_FRAMES):
    """Simulates a single car driven by the neural network on the track.

    Args:
        network: NeuralNetwork instance.
        track: Track instance.
        car_image: Pygame surface for the car.
        grid_slot_idx: Which START_GRID slot to place the car at.
        max_frames: Max frames to simulate.

    Returns:
        dict with simulation outcome (finished, finish_frame, progress, frames_alive, crashed).
    """
    gx, gy, g_angle = START_GRID[grid_slot_idx % len(START_GRID)]
    car = SpaceCar(gx, gy, car_image, name="gemini_trainee", angle=g_angle)

    while car.alive and car.frames_alive < max_frames and not car.finished:
        features = extract_features(car, track)
        outputs = network.forward(features)
        throttle_action, steer_action = interpret_outputs(outputs)
        apply_actions(car, throttle_action, steer_action)
        car.step(track)

    return {
        "finished": car.finished,
        "finish_frame": car.finish_frame if car.finish_frame is not None else max_frames,
        "progress": car.progress,
        "frames_alive": car.frames_alive,
        "crashed": car.crashed,
        "laps": car.laps,
    }


def compute_fitness(results):
    """Calculates a scalar fitness score across multiple grid evaluations.

    Scoring rules:
    - 1. Finishers are ranked by lap time (lower finish_frame = higher score).
    - 2. Non-finishers are ranked by progress.
    """
    total_score = 0.0
    for res in results:
        if res["finished"]:
            # Huge bonus for finishing, with higher score for faster lap times
            lap_time = res["finish_frame"]
            score = 100_000.0 + (ROUND_FRAMES - lap_time) * 50.0
        else:
            # Rank primarily by progress
            score = max(0.0, res["progress"])
            if res["crashed"]:
                score = max(0.0, score - 50.0)

        total_score += score

    return total_score / len(results)


def train(
    generations=100,
    pop_size=80,
    grid_slots=(0, 1),
    save_path="gemini_ai/brain.json",
    elite_count=8,
    mutation_rate=0.08,
    mutation_power=0.15,
):
    """Main neuroevolution training loop."""
    print("=" * 65)
    print(" GEMINI AI RACER - NEUROEVOLUTION TRAINER")
    print("=" * 65)
    print(f"Generations    : {generations}")
    print(f"Population size: {pop_size}")
    print(f"Grid slots     : {grid_slots}")
    print(f"Save path      : {save_path}")
    print("=" * 65)

    print("Loading track and assets...")
    track = load_track()
    car_surface = create_car_surface((0, 100, 255))
    print(f"Track loaded. Lap length: {track.lap_length} px.\n")

    # Initialize population
    population = []
    # If a pre-existing brain file exists, seed the population with it
    if os.path.exists(save_path):
        try:
            seed_net = NeuralNetwork.load(save_path)
            population.append(seed_net)
            print(f"Seeded population with existing brain from {save_path}")
        except Exception:
            pass

    while len(population) < pop_size:
        population.append(NeuralNetwork([11, 16, 12, 2]))

    best_overall_fitness = -1.0
    best_overall_net = None
    best_overall_lap_frame = float("inf")

    start_time = time.time()

    for gen in range(1, generations + 1):
        gen_start = time.time()

        # Evaluate entire population
        fitness_scores = []
        gen_finishers = 0
        gen_best_lap = float("inf")

        for net in population:
            results = []
            for slot in grid_slots:
                res = evaluate_car(net, track, car_surface, grid_slot_idx=slot)
                results.append(res)
                if res["finished"]:
                    gen_finishers += 1
                    if res["finish_frame"] < gen_best_lap:
                        gen_best_lap = res["finish_frame"]

            fit = compute_fitness(results)
            fitness_scores.append((fit, net, results))

        # Sort population by descending fitness
        fitness_scores.sort(key=lambda item: item[0], reverse=True)

        gen_best_fit = fitness_scores[0][0]
        gen_best_net = fitness_scores[0][1]
        gen_best_res = fitness_scores[0][2]

        avg_fit = sum(item[0] for item in fitness_scores) / pop_size
        elapsed = time.time() - gen_start

        # Check if improved overall
        is_new_best = gen_best_fit > best_overall_fitness
        if is_new_best:
            best_overall_fitness = gen_best_fit
            best_overall_net = gen_best_net.clone()
            best_overall_net.save(save_path)
            if gen_best_lap < best_overall_lap_frame:
                best_overall_lap_frame = gen_best_lap

        # Status output
        best_progress = max(r["progress"] for r in gen_best_res)
        status_msg = f"Gen {gen:3d}/{generations:3d} | Best Fit: {gen_best_fit:8.1f} | Avg Fit: {avg_fit:8.1f} | Best Prog: {best_progress:6.1f} px"
        if gen_best_lap < float("inf"):
            lap_sec = gen_best_lap / 60.0
            status_msg += f" | LAP FINISHED! Frame: {gen_best_lap} ({lap_sec:.2f}s)"
        if is_new_best:
            status_msg += " [*NEW BEST SAVED*]"
        status_msg += f" ({elapsed:.2f}s)"
        print(status_msg)

        if gen == generations:
            break

        # Selection and Breeding for next generation
        new_population = []

        # 1. Elitism: preserve top elites untouched
        for i in range(elite_count):
            new_population.append(fitness_scores[i][1].clone())

        # 2. Tournament selection function
        def select_parent(k=4):
            candidates = random.sample(fitness_scores[: pop_size // 2], min(k, pop_size // 2))
            candidates.sort(key=lambda c: c[0], reverse=True)
            return candidates[0][1]

        # 3. Create offspring via crossover and mutation
        while len(new_population) < pop_size:
            parent1 = select_parent()
            parent2 = select_parent()
            child = NeuralNetwork.crossover(parent1, parent2)
            child.mutate(rate=mutation_rate, power=mutation_power)
            new_population.append(child)

        population = new_population

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print(f" Total time: {total_time:.2f}s")
    print(f" Best fitness: {best_overall_fitness:.1f}")
    if best_overall_lap_frame < float("inf"):
        print(f" Fastest Lap : {best_overall_lap_frame} frames ({best_overall_lap_frame / 60.0:.2f}s)")
    print(f" Brain saved to: {save_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Gemini AI Racer using Neuroevolution")
    parser.add_argument("--generations", type=int, default=80, help="Number of generations to train")
    parser.add_argument("--pop-size", type=int, default=80, help="Population size per generation")
    parser.add_argument(
        "--slots",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated list of START_GRID slot indices to evaluate on (e.g. 0,1,2)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "brain.json"),
        help="Path to save the best brain JSON",
    )
    parser.add_argument("--elite-count", type=int, default=8, help="Number of elite networks to preserve")
    parser.add_argument("--mutation-rate", type=float, default=0.08, help="Probability of mutating each weight")
    parser.add_argument("--mutation-power", type=float, default=0.15, help="Standard deviation of mutations")

    args = parser.parse_args()
    slots = tuple(int(s.strip()) for s in args.slots.split(",") if s.strip())

    train(
        generations=args.generations,
        pop_size=args.pop_size,
        grid_slots=slots,
        save_path=args.save_path,
        elite_count=args.elite_count,
        mutation_rate=args.mutation_rate,
        mutation_power=args.mutation_power,
    )


if __name__ == "__main__":
    main()
