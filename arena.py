import sys
import pygame as pygame
from threading import Thread

BLACK = (50, 50, 50)
WHITE = (255, 255, 255)
GREEN = (0, 150, 136)
RED = (194, 24, 91)
BLUE = (0, 0, 128)
YELLOW = (255, 255, 0)
MARGIN = 2
BLOCK_SIZE = 20
PREY = "PREY"
PREDATOR = "PREDATOR"


def redistribute_rgb(r, g, b):
    threshold = 255.999
    m = max(r, g, b)
    if m <= threshold:
        return int(r), int(g), int(b)
    total = r + g + b
    if total >= 3 * threshold:
        return int(threshold), int(threshold), int(threshold)
    x = (3 * threshold - total) / (3 * m - total)
    gray = threshold - x * m
    return int(gray + x * r), int(gray + x * g), int(gray + x * b)

class Env:
    def __init__(self, grid, enableUi):
        self.grid = grid
        self.screen = None

        if enableUi:
            pygame.init()
            window_size = grid.numberOfColumns * BLOCK_SIZE + (MARGIN * grid.numberOfRows + 1)
            self.screen = pygame.display.set_mode((window_size, window_size))
            self.screen.fill(BLACK)

            t = Thread(target=self.render_loop)
            t.start()
            self.draw_grid()

    def update_grid(self):
        self.draw_grid()

    @staticmethod
    def render_loop():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def draw_grid(self):

        grid = self.grid.grid
        for x in range(0, self.grid.numberOfRows):
            for y in range(0, self.grid.numberOfColumns):
                # draw prey
                color = WHITE
                if grid[y, x, 2] != 0:
                    id = 1 + int(grid[y][x][2]) * -0.5
                    color = redistribute_rgb(127 / id, 255 / id, 0 / id)

                pygame.draw.rect(self.screen, color, [(MARGIN + BLOCK_SIZE) * x + MARGIN,
                                                      (MARGIN + BLOCK_SIZE) * y + MARGIN,
                                                      BLOCK_SIZE,
                                                      BLOCK_SIZE])
                # draw predator
                if grid[y, x, 3] != 0:
                    if grid[y, x, 3] == 2 or grid[y, x, 3] == 3 or\
                            grid[y, x, 3] == 4 or grid[y, x, 3] == 5:
                        pygame.draw.circle(self.screen, RED, ((MARGIN + BLOCK_SIZE) * x + MARGIN + BLOCK_SIZE / 2,
                                                              (MARGIN + BLOCK_SIZE) * y + MARGIN + BLOCK_SIZE / 2),
                                           BLOCK_SIZE / 2.5)
                    elif grid[y, x, 3] == 6 or grid[y, x, 3] == 7 or\
                            grid[y, x, 3] == 8 or grid[y, x, 3] == 9:
                        pygame.draw.circle(self.screen, BLUE, ((MARGIN + BLOCK_SIZE) * x + MARGIN + BLOCK_SIZE / 2,
                                                              (MARGIN + BLOCK_SIZE) * y + MARGIN + BLOCK_SIZE / 2),
                                           BLOCK_SIZE / 2.5)
                    else:
                        pygame.draw.circle(self.screen, GREEN, ((MARGIN + BLOCK_SIZE) * x + MARGIN + BLOCK_SIZE / 2,
                                                              (MARGIN + BLOCK_SIZE) * y + MARGIN + BLOCK_SIZE / 2),
                                           BLOCK_SIZE / 2.5)

        pygame.display.update()