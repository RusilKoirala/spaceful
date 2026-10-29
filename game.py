# Shared game setup: used by main.py (play it yourself) and by every AI's trainer.
# No AI code lives here.
import pygame
from settings import *
pygame.init()
from car import *
from track import *

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)


def load_track():
    track_img = pygame.image.load("assets/track.png").convert_alpha()
    track_border_img = pygame.image.load("assets/track_border.png").convert_alpha()
    return Track(track_img, track_border_img)


def create_car_surface(color):
    surface = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
    surface.fill(color)
    return surface
