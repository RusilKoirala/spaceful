# Train every AI side by side in one window, under the same rules.
#   uv run python arena.py
#  
# uv run python arena.py --population 40 --generations 50
#   uv run python arena.py --population 40 --generations 50 --resume   (continue from saved brains)
# Each AI keeps its own network, driver, crossover and mutation code. The arena only
# shares the environment: population size, elites, parent pool, mutation, start slot,
# frame limit, fitness, and every generation starts and ends at the same time for all.
# Everyone starts from random weights unless --resume is given, which continues from the
# brains last saved in arena_brains/ (the rest of the population is bred from them).
#
# Keys: UP / DOWN = simulation speed, R = toggle leader rays
import argparse
import json
import os
import pickle
import random

from game import *

import rusil_ai.config as rusil_config
from rusil_ai.network import NeuralNetwork as RusilNetwork, breed as rusil_breed
from rusil_ai.agent import drive as rusil_drive
from claude_ai.network import Network as ClaudeNetwork, apply as claude_apply
from claude_ai.train import crossover as claude_crossover, mutate as claude_mutate
from chatgpt_ai.brain import Brain as ChatGPTBrain, LAYERS as CHATGPT_LAYERS, WEIGHT_COUNT as CHATGPT_WEIGHTS, control as chatgpt_control
from gemini_ai.model import NeuralNetwork as GeminiNetwork, apply_action as gemini_apply

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_brains")
SCALE = 0.5                       # each AI gets a quarter of the window
QW, QH = WIDTH // 2, HEIGHT // 2


def fitness(car):
    # same for everyone, matches the race: finishing beats not finishing, faster lap beats slower
    if car.finished:
        return 100000 + (ROUND_FRAMES - car.finish_frame) * 10
    return max(car.progress, 0)


# ---- one adapter per AI: how to make, breed, mutate, drive and save its brain ----

class Rusil:
    name, color = "Rusil", (255, 0, 0)
    file = "rusil_breeding_pool.pkl"

    def new(self):
        return RusilNetwork(rusil_config.BRAIN_SIZES)

    def crossover(self, a, b):
        return rusil_breed(a, b)

    def mutate(self, brain, rate):
        brain.mutate(rate)
        return brain

    def drive(self, car, track):
        rusil_drive(car, track)   # reads car.brain

    def save(self, brains, path):
        with open(path, "wb") as f:
            pickle.dump(brains, f)   # same breeding-pool format rusil_ai.train loads

    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ChatGPT:
    name, color = "ChatGPT", (255, 255, 255)
    file = "chatgpt_brain.json"

    def new(self):
        return ChatGPTBrain([random.gauss(0, 0.5) for _ in range(CHATGPT_WEIGHTS)])

    def crossover(self, a, b):
        return ChatGPTBrain([x if random.random() < 0.5 else y for x, y in zip(a.weights, b.weights)])

    def mutate(self, brain, rate):
        return ChatGPTBrain([v + random.gauss(0, 0.18) if random.random() < rate else v
                             for v in brain.weights])

    def drive(self, car, track):
        chatgpt_control(car.brain, car, track, RAY_LENGTH, MAX_SPEED)

    def save(self, brains, path):
        with open(path, "w") as f:
            # same format chatgpt_ai.race loads
            json.dump({"source": "chatgpt_ai.train", "layers": list(CHATGPT_LAYERS),
                       "weights": list(brains[0].weights),
                       "normalization": {"ray_length": RAY_LENGTH, "max_speed": MAX_SPEED}}, f, indent=2)

    def load(self, path):
        return [ChatGPTBrain.load(path)]


class Gemini:
    name, color = "Gemini", (0, 0, 255)
    file = "gemini_brain.json"

    def new(self):
        return GeminiNetwork()

    def crossover(self, a, b):
        return GeminiNetwork.crossover(a, b)

    def mutate(self, brain, rate):
        brain.mutate(rate=rate, mutation_strength=0.1)
        return brain

    def drive(self, car, track):
        inputs = [d / RAY_LENGTH for d in car.get_distances(track)] + [car.speed / MAX_SPEED]
        gemini_apply(car, car.brain.forward(inputs))

    def save(self, brains, path):
        brains[0].save(path)

    def load(self, path):
        brain = GeminiNetwork()
        brain.load(path)
        return [brain]


class Claude:
    name, color = "Claude", (255, 165, 0)
    file = "claude_brain.json"

    def new(self):
        return ClaudeNetwork()

    def crossover(self, a, b):
        return ClaudeNetwork(claude_crossover(a.genome, b.genome))

    def mutate(self, brain, rate):
        return ClaudeNetwork(claude_mutate(brain.genome, rate, 0.1))

    def drive(self, car, track):
        claude_apply(car, *car.brain.decide(car.get_distances(track), car.speed))

    def save(self, brains, path):
        brains[0].save(path)

    def load(self, path):
        return [ClaudeNetwork.load(path)]


# ---- one team = one AI's population ----

class Team:
    def __init__(self, ai, args):
        self.ai = ai
        self.args = args
        self.image = create_car_surface(ai.color)
        small = (round(CAR_SIZE[0] * SCALE), round(CAR_SIZE[1] * SCALE))
        self.small_image = pygame.transform.scale(self.image, small)
        self.path = os.path.join(SAVE_DIR, ai.file)
        self.brains = self.load_saved() if args.resume else []
        self.brains += [ai.new() for _ in range(args.population - len(self.brains))]

        self.best_score = -1
        self.best_lap = None
        self.best_lap_gen = None
        self.last = "-"
        self.spawn()

    def load_saved(self):
        # keep the saved brains as-is, breed the rest of the population from them
        if not os.path.exists(self.path):
            print(f"{self.ai.name}: no saved brain, starting from random weights")
            return []
        saved = self.ai.load(self.path)[:self.args.population]
        brains = list(saved)
        while len(brains) < self.args.population:
            child = self.ai.crossover(random.choice(saved), random.choice(saved))
            brains.append(self.ai.mutate(child, self.args.mutation))
        print(f"{self.ai.name}: resumed from {len(saved)} saved brain(s)")
        return brains

    def spawn(self):
        x, y, angle = START_GRID[self.args.slot]
        self.cars = []
        for i, brain in enumerate(self.brains):
            car = SpaceCar(x, y, self.image, f"{self.ai.name} {i}", angle)
            car.brain = brain
            self.cars.append(car)

    def any_alive(self):
        return any(car.alive for car in self.cars)

    def step(self, track):
        for car in self.cars:
            if car.alive:
                self.ai.drive(car, track)
                car.step(track)

    def leader(self):
        alive = [car for car in self.cars if car.alive]
        return max(alive or self.cars, key=fitness)

    def evolve(self, gen, track):
        ranked = sorted(self.cars, key=fitness, reverse=True)
        best = ranked[0]
        score = fitness(best)
        brains = [car.brain for car in ranked]

        finishers = [car for car in self.cars if car.finished]
        if finishers:
            fastest = min(car.finish_frame for car in finishers)
            if self.best_lap is None or fastest < self.best_lap:
                self.best_lap, self.best_lap_gen = fastest, gen
            self.last = f"{len(finishers)} finished, fastest {fastest / FPS:.2f}s"
        else:
            self.last = f"best progress {max(best.progress, 0):.0f}/{track.lap_length}"

        if score > self.best_score:
            self.best_score = score
            self.ai.save(brains[:self.args.elite], self.path)

        # same breeding rules for everyone: keep the elites, breed the rest from the top parents
        parents = brains[:self.args.parents]
        self.brains = brains[:self.args.elite]
        while len(self.brains) < self.args.population:
            mom, dad = random.choice(parents), random.choice(parents)
            child = self.ai.crossover(mom, dad)
            self.brains.append(self.ai.mutate(child, self.args.mutation))
        self.spawn()


# ---- drawing ----

big_font = pygame.font.SysFont(None, 24)


def text_color_on(color):
    r, g, b = color
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "white"


def draw_team(team, ox, oy, small_track, track, gen, show_rays):
    screen.blit(small_track, (ox, oy))
    leader = team.leader()

    for car in team.cars:
        if not car.alive:
            continue
        img = pygame.transform.rotate(team.small_image, car.angle)
        screen.blit(img, img.get_rect(center=(ox + car.x * SCALE, oy + car.y * SCALE)))

    if show_rays and leader.alive:
        cx, cy = ox + leader.x * SCALE, oy + leader.y * SCALE
        for offset in OFFSETS:
            px, py, _ = leader.cast_ray(track, offset)
            end = (ox + px * SCALE, oy + py * SCALE)
            pygame.draw.line(screen, team.ai.color, (cx, cy), end)
            pygame.draw.circle(screen, team.ai.color, end, 2)

    pygame.draw.rect(screen, team.ai.color, (ox, oy, QW, QH), 2)

    # colored tab in the corner with the AI's name
    label = big_font.render(f"{team.ai.name}  gen {gen}", True, text_color_on(team.ai.color))
    tab = pygame.Rect(ox, oy, label.get_width() + 16, label.get_height() + 8)
    pygame.draw.rect(screen, team.ai.color, tab, border_bottom_right_radius=8)
    screen.blit(label, (ox + 8, oy + 4))

    alive = sum(car.alive for car in team.cars)
    finished = sum(car.finished for car in team.cars)
    best_lap = f"{team.best_lap / FPS:.2f}s (gen {team.best_lap_gen})" if team.best_lap else "-"
    lines = [
        f"Alive {alive}/{len(team.cars)}   Finished {finished}",
        f"Leader progress {max(leader.progress, 0):.0f}/{track.lap_length}",
        f"Best lap {best_lap}",
        f"Last gen: {team.last}",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, "white"), (ox + 8, oy + tab.height + 4 + i * 15))


def draw(teams, small_track, track, gen, frame, speed, show_rays):
    screen.fill("black")
    for i, team in enumerate(teams):
        draw_team(team, (i % 2) * QW, (i // 2) * QH, small_track, track, gen, show_rays)

    hud = font.render(f"Gen {gen}   frame {frame}/{ROUND_FRAMES}   speed x{speed}   "
                      f"[UP/DOWN speed, R rays]", True, "white")
    box = hud.get_rect(midbottom=(WIDTH // 2, HEIGHT - 6)).inflate(16, 8)
    pygame.draw.rect(screen, "black", box)
    pygame.draw.rect(screen, "gray40", box, 1)
    screen.blit(hud, hud.get_rect(center=box.center))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Train all AIs in the same environment")
    parser.add_argument("--population", type=int, default=30, help="cars per AI")
    parser.add_argument("--elite", type=int, default=4, help="best brains kept unchanged each generation")
    parser.add_argument("--parents", type=int, default=None, help="top brains allowed to breed (default population/4)")
    parser.add_argument("--mutation", type=float, default=0.1, help="mutation rate passed to every AI")
    parser.add_argument("--generations", type=int, default=0, help="stop after this many (0 = run until closed)")
    parser.add_argument("--resume", action="store_true", help="continue from the brains saved in arena_brains/")
    parser.add_argument("--slot", type=int, default=0, help="START_GRID slot every car starts from")
    args = parser.parse_args()
    args.parents = max(2, args.parents or args.population // 4)
    args.elite = min(args.elite, args.population)

    os.makedirs(SAVE_DIR, exist_ok=True)
    track = load_track()
    w, h = track.image.get_size()
    small_track = pygame.transform.smoothscale(track.image, (round(w * SCALE), round(h * SCALE)))
    pygame.display.set_caption("Spaceful arena")

    teams = [Team(ai, args) for ai in (Rusil(), ChatGPT(), Gemini(), Claude())]
    gen, frame, speed = 1, 0, 1
    show_rays = True
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    speed = min(speed * 2, 64)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed // 2, 1)
                elif event.key == pygame.K_r:
                    show_rays = not show_rays

        for _ in range(speed):
            for team in teams:
                team.step(track)
            frame += 1
            if frame >= ROUND_FRAMES or not any(team.any_alive() for team in teams):
                print(f"Generation {gen}:")
                for team in teams:
                    team.evolve(gen, track)
                    print(f"  {team.ai.name:8s} {team.last}")
                gen += 1
                frame = 0
                if args.generations and gen > args.generations:
                    running = False
                break

        draw(teams, small_track, track, gen, frame, speed, show_rays)
        clock.tick(FPS if speed == 1 else 0)

    pygame.quit()
    print(f"Best brains saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()
