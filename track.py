# track.py
import pygame

class Track:
    def __init__(self, visual_image, border_image):
        self.image = visual_image
        self.border_image = border_image
       
        self.mask = pygame.mask.from_surface(self.border_image)
        self.x = 0
        self.y = 0

    def draw(self, win):
        win.blit(self.image, (self.x, self.y))