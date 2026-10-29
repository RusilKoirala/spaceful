# Trains the Claude racer with neuroevolution (genetic algorithm over network weights).
# Run from the project root:  uv run python -m claude_ai.train
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import multiprocessing as mp
import random
import sys
import time

from claude_ai.network import Network, apply, n_weights

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN_PATH = os.path.join(HERE, "brain.json")

POPULATION = 80
ELITE = 8
GENERATIONS = int(os.environ.get("GENERATIONS", 150))
EVAL_SLOTS = [0, 1, 3, 5]   # grid slots used to score each genome (both lanes, front and back)

_track = None
_image = None


def _worker_init():
    global _track, _image
    import game
    _track = game.load_track()
    _image = game.create_car_surface((255, 140, 0))


def run_car(net, slot):
    import game
    x, y, angle = game.START_GRID[slot]
    car = game.SpaceCar(x, y, _image, "claude", angle)
    for _ in range(game.ROUND_FRAMES):
        throttle, steer = net.decide(car.get_distances(_track), car.speed)
        apply(car, throttle, steer)
        car.step(_track)
        if not car.alive:
            break
    if car.finished:
        # finishing always beats not finishing; faster lap beats slower lap
        return 100000 + (game.ROUND_FRAMES - car.finish_frame) * 10, car.finish_frame
    return car.progress, None


def evaluate(genome):
    net = Network(genome)
    scores, frames = [], []
    for slot in EVAL_SLOTS:
        s, f = run_car(net, slot)
        scores.append(s)
        frames.append(f)
    # reward the average but care about the worst slot too, so it is robust on any grid slot
    return 0.5 * sum(scores) / len(scores) + 0.5 * min(scores), frames


def mutate(genome, rate, power):
    return [g + random.gauss(0, power) if random.random() < rate else g for g in genome]


def crossover(a, b):
    return [x if random.random() < 0.5 else y for x, y in zip(a, b)]


def main():
    random.seed(int(os.environ.get("SEED", time.time())))
    size = n_weights()
    population = [Network().genome for _ in range(POPULATION)]
    if os.path.exists(BRAIN_PATH) and os.environ.get("FRESH") != "1":
        best = Network.load(BRAIN_PATH).genome
        if len(best) == size:
            print("continuing from saved brain")
            population[0] = best
            population[1:POPULATION // 2] = [mutate(best, 0.2, 0.2) for _ in range(POPULATION // 2 - 1)]

    best_ever = None
    with mp.Pool(initializer=_worker_init) as pool:
        for gen in range(GENERATIONS):
            t = time.time()
            results = pool.map(evaluate, population)
            ranked = sorted(zip(results, population), key=lambda r: r[0][0], reverse=True)
            (top_score, top_frames), top = ranked[0]

            if best_ever is None or top_score > best_ever[0]:
                best_ever = (top_score, top)
                Network(top).save(BRAIN_PATH)

            print(f"gen {gen:3d}  best {top_score:9.1f}  frames {top_frames}  "
                  f"({time.time() - t:.1f}s)", flush=True)

            # next generation: keep elites, breed the rest from the top quarter
            parents = [g for _, g in ranked[:POPULATION // 4]]
            population = [g for _, g in ranked[:ELITE]]
            while len(population) < POPULATION:
                a, b = random.choice(parents), random.choice(parents)
                child = crossover(a, b) if random.random() < 0.5 else list(a)
                power = random.choice((0.03, 0.1, 0.3))
                population.append(mutate(child, random.choice((0.05, 0.15)), power))

    print(f"saved best brain to {BRAIN_PATH} (score {best_ever[0]:.1f})")


if __name__ == "__main__":
    sys.exit(main())
