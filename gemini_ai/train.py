import os
import sys
import time
import random

# Headless training: must set SDL_VIDEODRIVER before importing game
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Ensure package root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from game import (
    load_track,
    create_car_surface,
    SpaceCar,
    START_GRID,
    RAY_LENGTH,
    MAX_SPEED,
    ROUND_FRAMES,
    STALL_FRAMES,
    MIN_PROGRESS,
)

try:
    from .model import NeuralNetwork, apply_action
except ImportError:
    from gemini_ai.model import NeuralNetwork, apply_action


def evaluate_population(population, track, car_img, grid_slot_idx=0):
    """
    Evaluates all individuals in the population simultaneously in the game environment.
    Cars do not collide with each other, so evaluating them in parallel is fast.
    """
    gx, gy, g_angle = START_GRID[grid_slot_idx % len(START_GRID)]

    cars = [
        SpaceCar(gx, gy, car_img, f"gemini_{i}", g_angle)
        for i in range(len(population))
    ]

    for frame in range(ROUND_FRAMES):
        active_count = 0
        for net, car in zip(population, cars):
            if not car.alive or car.finished:
                continue

            active_count += 1
            distances = car.get_distances(track)
            inputs = [d / RAY_LENGTH for d in distances] + [car.speed / MAX_SPEED]
            outputs = net.forward(inputs)
            apply_action(car, outputs)
            car.step(track)

        if active_count == 0:
            break

    # Calculate fitness strictly following the race scoring rules:
    # 1. Finishers are ranked by lap time (finish_frame).
    # 2. Non-finishers are ranked by progress.
    scores = []
    for net, car in zip(population, cars):
        if car.finished:
            # Finishing bonus + bonus for faster lap times
            fitness = 100000.0 + (ROUND_FRAMES - car.finish_frame) * 50.0
        else:
            # Maximizing forward track progress
            fitness = max(0.0, float(car.progress))
        scores.append((fitness, car.finished, car.finish_frame, float(car.progress)))

    return scores


def tournament_select(population, scores, k=4):
    """Select the best individual from a random tournament of size k."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: scores[i][0])
    return population[best_idx]


def train(generations=150, pop_size=60, elite_count=6):
    track = load_track()
    car_img = create_car_surface((0, 210, 255))  # Gemini electric cyan

    save_path = os.path.join(os.path.dirname(__file__), "brain.json")

    # Initial population with weights sampled from a single Gaussian distribution
    population = [NeuralNetwork(init_sigma=0.5) for _ in range(pop_size)]

    best_overall_fitness = -1.0
    best_overall_net = None
    best_finish_time = None

    print("=" * 65)
    print("🚀 Starting Gemini AI Evolutionary Training")
    print(f"Population: {pop_size} | Elites: {elite_count} | Generations: {generations}")
    print(f"Brain save path: {save_path}")
    print("=" * 65)

    try:
        for gen in range(1, generations + 1):
            start_t = time.time()
            grid_slot = (gen - 1) % len(START_GRID)
            scores = evaluate_population(population, track, car_img, grid_slot_idx=grid_slot)

            # Sort population by fitness descending
            ranked = sorted(
                zip(population, scores),
                key=lambda pair: pair[1][0],
                reverse=True,
            )

            gen_best_net, (gen_best_fit, gen_finished, gen_time, gen_prog) = ranked[0]
            avg_fit = sum(s[0] for s in scores) / len(scores)
            elapsed = time.time() - start_t

            # Update overall best
            if gen_best_fit > best_overall_fitness:
                best_overall_fitness = gen_best_fit
                best_overall_net = gen_best_net.clone()
                best_overall_net.save(save_path)
                if gen_finished:
                    best_finish_time = gen_time
                status_update = " ⭐ (NEW BEST SAVED)"
            else:
                status_update = ""

            # Format log line
            if gen_finished:
                res_str = f"🏁 FINISHED in {gen_time} frames ({gen_time / 60.0:.2f}s)"
            else:
                res_str = f"progress: {gen_prog:6.1f}px"

            print(
                f"Gen {gen:3d}/{generations} | {res_str} | "
                f"Best Fit: {gen_best_fit:8.1f} | Avg: {avg_fit:6.1f} | "
                f"{elapsed:4.2f}s{status_update}"
            )

            # Build next generation
            next_population = []

            # 1. Elitism: preserve top performers
            for i in range(elite_count):
                next_population.append(ranked[i][0].clone())

            # 2. Reproduction via Crossover and Mutation
            while len(next_population) < pop_size:
                r = random.random()
                if r < 0.70:
                    # Crossover + mutation
                    p1 = tournament_select(population, scores, k=4)
                    p2 = tournament_select(population, scores, k=4)
                    child = NeuralNetwork.crossover(p1, p2)
                    child.mutate(rate=0.15, mutation_strength=0.10)
                    next_population.append(child)
                elif r < 0.95:
                    # Single-parent mutation
                    parent = tournament_select(population, scores, k=4)
                    child = parent.clone()
                    child.mutate(rate=0.15, mutation_strength=0.10)
                    next_population.append(child)
                else:
                    # Genetic diversity: fresh immigrant
                    next_population.append(NeuralNetwork(init_sigma=0.5))

            population = next_population

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    print("=" * 65)
    print("🏆 Training session completed!")
    if best_finish_time is not None:
        print(f"Best Lap Time: {best_finish_time} frames ({best_finish_time / 60.0:.2f}s)")
    print(f"Best Fitness: {best_overall_fitness:.1f}")
    if best_overall_net is not None:
        best_overall_net.save(save_path)
        print(f"Trained brain saved to: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    train()

