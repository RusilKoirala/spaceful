# track.py
import pygame
from collections import deque
from array import array
from settings import START_GRID


class Track:
    def __init__(self, visual_image, border_image):
        self.image = visual_image
        self.border_image = border_image

        self.mask = pygame.mask.from_surface(self.border_image)
        self.x = 0
        self.y = 0

        self.build_progress_map()

    def draw(self, win):
        win.blit(self.image, (self.x, self.y))

    def build_progress_map(self):
        # For every road pixel: how many pixels you must drive from the start line,
        # going the right way (start facing right), to reach it. Walls are -1.
        w, h = self.mask.get_size()
        self.width, self.height = w, h
        self.progress = array("i", [-1]) * (w * h)

        start_x, start_y, _ = START_GRID[0]
        start_x, start_y = int(start_x), int(start_y)
        # A wall just behind the start line, only for this search, so it has to
        # go the long way round the track to reach the pixels behind the start
        barrier_x = start_x - 15

        def is_road(x, y):
            return 0 <= x < w and 0 <= y < h and not self.mask.get_at((x, y))

        self.progress[start_y * w + start_x] = 0
        queue = deque([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            d = self.progress[y * w + x] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if nx == barrier_x and abs(ny - start_y) < 100:
                    continue
                if is_road(nx, ny) and self.progress[ny * w + nx] == -1:
                    self.progress[ny * w + nx] = d
                    queue.append((nx, ny))

        self.lap_length = max(self.progress)

    def progress_at(self, x, y):
        x, y = int(x - self.x), int(y - self.y)
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.progress[y * self.width + x]
        return -1
