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
        self.crashed = False
        self.finished = False
        self.finish_frame = None
        self.frames_alive = 0
        
        self.progress = 0
        self.laps = 0
        self.last_spot = None
        self.check_progress = 0
       
        self.distance = 0

    def get_rotated(self):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        return rotated_image, rect

    def draw(self, win, track, show_rays=True):
        rotated_image, rect = self.get_rotated()
        if not self.alive and not self.finished:
            rotated_image.set_alpha(80)
        win.blit(rotated_image, rect.topleft)

        if not self.alive or not show_rays:
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
        self.update_progress(track)

        if self.laps >= LAPS:
            # crossed the finish line: park the car, its race is done
            self.finished = True
            self.finish_frame = self.frames_alive
            self.alive = False
            self.speed = 0
            return

        if self.collide(track.mask, track.x, track.y):
            self.crashed = True
            self.alive = False
            self.speed = 0
        elif self.frames_alive % STALL_FRAMES == 0:
            if self.progress - self.check_progress < MIN_PROGRESS:
                self.alive = False
                self.speed = 0
            self.check_progress = self.progress

    def update_progress(self, track):
        spot = track.progress_at(self.x, self.y)
        if spot < 0:
            return
        if self.last_spot is None:
            self.last_spot = spot

        change = spot - self.last_spot
        if change < -track.lap_length / 2:
            # jumped from the end of the lap back to the start: finished a lap
            change += track.lap_length
            self.laps += 1
        elif change > track.lap_length / 2:
            # jumped from the start back to the end: crossed the line going backwards
            change -= track.lap_length
            self.laps -= 1

        self.progress += change
        self.last_spot = spot

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
