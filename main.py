
import pygame
from settings import *
pygame.init()
from car import *
from track import *


screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)

track_img = pygame.image.load("assets/track.png").convert_alpha()
track_border_img = pygame.image.load("assets/track_border.png").convert_alpha()


def create_car_surface(color):
    surface = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
    surface.fill(color)
    return surface


def idle(car):
    car.reduce_speed()


def keyboard(car):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a]:
        car.rotate(left=True)
    if keys[pygame.K_d]:
        car.rotate(right=True)

    if keys[pygame.K_w]:
        car.move_forward()
    elif keys[pygame.K_s]:
        car.move_backward()
    else:
        car.reduce_speed()



AGENTS = [
    ("Rusil",   (255, 0, 0),     keyboard if TEST else idle)
   # ("ChatGPT", (255, 255, 255), idle),
   # ("Gemini",  (0, 0, 255),     idle),
   # ("Claude",  (255, 165, 0),   idle),
]


def spawn_agents(agents):
    cars = []
    for i, (name, color, controller) in enumerate(agents):
        x, y, angle = START_GRID[i % len(START_GRID)]
        car = SpaceCar(x, y, create_car_surface(color), name, angle)
        car.controller = controller
        cars.append(car)
    return cars


def draw(screen, cars, track, round_no):
    screen.fill("black")
    track.draw(screen)

    for car in cars:
        car.draw(screen,track)

    screen.blit(font.render(f"Round {round_no}", True, "white"), (10, 10))
    for i, car in enumerate(cars):
        status = f"speed {car.speed:.1f}" if car.alive else "CRASHED"
        label = font.render(f"{car.name}  {status}  distance {car.distance:.0f}", True, "white")
        screen.blit(label, (10, 26 + i * 16))

    pygame.display.flip()


def main():
    running = True
    track = Track(track_img,track_border_img)
    cars = spawn_agents(AGENTS)
    round_no = 1
    round_frames = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for car in cars:
            if car.alive:
                car.controller(car)
                car.step(track)
                if TEST:
                    print(car.name, car.get_distances(track))

        
        round_frames += 1
        if not any(car.alive for car in cars) or round_frames >= ROUND_FRAMES:
            cars = spawn_agents(AGENTS)
            round_no += 1
            round_frames = 0

        draw(screen,cars,track,round_no)
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
