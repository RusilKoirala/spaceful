import pygame
import math
from settings import *
from utils import *


class SpaceCar:

    def __init__(self, x, y, image, name="car", angle=0):
        self.image = image
        self.name = name
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.alive = True
        self.frames_alive = 0
        self.stall_frames = 0
        # Score: total forward distance driven (reversing doesn't count)
        self.distance = 0

    def get_rotated(self):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        return rotated_image, rect

    def draw(self, win, track):
        rotated_image, rect = self.get_rotated()
        if not self.alive:
            rotated_image.set_alpha(80)
        win.blit(rotated_image, rect.topleft)

        if not self.alive:
            return

        for offset in OFFSETS:
            cx,cy= self.x,self.y
            px,py,distance = self.cast_ray(track,offset)
            pygame.draw.circle(win,"yellow",(int(px),int(py)),3)
            pygame.draw.line(win,"yellow",(cx,cy),(int(px),int(py)))

    def move_forward(self):
        self.speed = min(self.speed + ACCELERATION, MAX_SPEED)

    def move_backward(self):
        self.speed = max(self.speed - ACCELERATION, -MAX_SPEED / 2)

    def rotate(self, left=False, right=False):
        if left:
            self.angle += ROTATION_SPEED
        elif right:
            self.angle -= ROTATION_SPEED

    def update_position(self):
        radians = math.radians(self.angle)

        self.x += math.cos(radians) * self.speed
        self.y -= math.sin(radians) * self.speed

    def reduce_speed(self):
        if self.speed > 0:
            self.speed = max(self.speed - FRICTION, 0)
        elif self.speed < 0:
            self.speed = min(self.speed + FRICTION, 0)

    def collide(self, mask, x=0, y=0):
        rotated_image, rect = self.get_rotated()
        car_mask = pygame.mask.from_surface(rotated_image)
        offset = (int(rect.left - x), int(rect.top - y))
        return mask.overlap(car_mask, offset)

    def step(self, track):
    
        if not self.alive:
            return
        self.update_position()
        self.frames_alive += 1
        self.distance += max(self.speed, 0)

        if self.speed < STALL_SPEED:
            self.stall_frames += 1
        else:
            self.stall_frames = 0

        if self.stall_frames >= STALL_FRAMES or self.collide(track.mask, track.x, track.y):
            self.alive = False
            self.speed = 0

    def cast_ray(self, track, offset):
        radians = math.radians(self.angle+offset)
        w, h = track.mask.get_size()
        distance = 0

      
        while distance < RAY_LENGTH:
            px = self.x+math.cos(radians)*distance
            py = self.y-math.sin(radians)*distance

            mx, my = int(px - track.x), int(py - track.y)
            if mx < 0 or my < 0 or mx >= w or my >= h:
                break
            if track.mask.get_at((mx, my)):
                break

            distance += RAY_STEP

        return px,py,distance

    def get_distances(self, track):
       
        return [self.cast_ray(track, offset)[2] for offset in OFFSETS]
