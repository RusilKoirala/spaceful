
#   uv run python -m rusil_ai.train
import copy
import os
import pickle
import random

from game import *
from rusil_ai.config import *
from rusil_ai.network import NeuralNetwork, breed
from rusil_ai.agent import drive


def fitness(car):
    
    # progress only counts driving the right way round the track
    progress = max(car.progress, 0)
    avg_speed = progress / car.frames_alive if car.frames_alive else 0
   
    score = progress * (1 + avg_speed / MAX_SPEED)
    if car.crashed:
        score *= CRASH_PENALTY
    if car.finished:
        # faster finish = higher average speed = higher score already, plus a bonus for finishing at all
        score *= FINISH_BONUS
    return score


def spawn_population(pool):
    # The breeding pool (last round's best cars) drive again unchanged (red).
    # Everyone else is a child of two random parents from the pool, plus a little mutation (blue).
    cars = []
    x, y, angle = START_GRID[0]
    for i in range(POPULATION):
        if pool is None:
            brain = NeuralNetwork(BRAIN_SIZES)
        elif i < len(pool):
            brain = copy.deepcopy(pool[i])
        else:
            mom, dad = random.sample(pool, 2)
            brain = breed(mom, dad)
            brain.mutate(MUTATION)

        color = (255, 0, 0) if pool is not None and i < len(pool) else (0, 200, 255)
        car = SpaceCar(x, y, create_car_surface(color), f"Rusil {i}", angle)
        car.controller = drive
        car.brain = brain
        cars.append(car)
    return cars


def load_pool():
    if os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, "rb") as f:
            return pickle.load(f)
    return None


def save_pool(pool):
    with open(BRAIN_FILE, "wb") as f:
        pickle.dump(pool, f)


def draw(screen, cars, track, round_no, best_score, best_lap):
    screen.fill("black")
    track.draw(screen)

    leader = max(cars, key=fitness)
    for car in cars:
        car.draw(screen, track, show_rays=(car is leader))

    alive = sum(car.alive for car in cars)
    finished = sum(car.finished for car in cars)
    lines = [
        f"Rusil AI - round {round_no}",
        f"Alive {alive}/{len(cars)}   Finished {finished}",
        f"Best lap {best_lap / FPS:.2f}s" if best_lap else "Best lap -",
        f"Leader score {fitness(leader):.0f}",
        f"Last round best {best_score:.0f}",
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, "white"), (10, 10 + i * 16))

    pygame.display.flip()


def main():
    running = True
    track = load_track()
    pool = load_pool()
    best_score = 0
    best_lap = None   # fewest frames any car needed to finish
    cars = spawn_population(pool)
    round_no = 1
    round_frames = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for car in cars:
            if car.alive:
                car.controller(car, track)
                car.step(track)

        round_frames += 1
        if not any(car.alive for car in cars) or round_frames >= ROUND_FRAMES:
            ranked = sorted(cars, key=fitness, reverse=True)
            pool = [car.brain for car in ranked[:BREEDING_POOL]]
            best = ranked[0]
            best_score = fitness(best)
            save_pool(pool)
            print(f"Round {round_no}: best score {best_score:.0f}  progress {best.progress:.0f}/{track.lap_length}  laps {best.laps}  crashed {best.crashed}")

            finishers = [car for car in cars if car.finished]
            if finishers:
                fastest = min(car.finish_frame for car in finishers)
                if best_lap is None or fastest < best_lap:
                    best_lap = fastest
                print(f"  FINISHED: {len(finishers)} cars crossed the line, fastest {fastest / FPS:.2f}s (best ever {best_lap / FPS:.2f}s)")
                if STOP_WHEN_FINISHED:
                    print(f"Rusil AI finished the race in round {round_no}!")
                    running = False

            cars = spawn_population(pool)
            round_no += 1
            round_frames = 0

        draw(screen, cars, track, round_no, best_score, best_lap)
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
