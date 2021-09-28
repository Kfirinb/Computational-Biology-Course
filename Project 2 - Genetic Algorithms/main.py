"""
Computational Biology: Second Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
May 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

# Basic grid design, courtesy of Auctux: https://www.youtube.com/watch?v=d73z8U0iUYE

import pygame
import sys
import os
import gridRefactor as grid
import numpy as np

os.environ["SDL_VIDEO_CENTERED"]='1'

width, height = 500,500 #792, 792
size = (width, height)

pygame.init()
pygame.display.set_caption("Computational Biology - Assignment 2 By CUCUMBER and Kfir Inbal")
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
fps = 60

black = (0, 0, 0)
blue = (0, 121, 150)
blue1 = (0,14,71)
white = (255, 255, 255)

# scaler = 30
offset = 1

Grid = grid.Grid(width, height, offset, screen)
#Grid.random2d_array(1)
Grid.random2d_array()

run = True
while run:
    clock.tick(fps)
    screen.fill(black)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    Grid.gridUpdater(off_color=white, on_color=blue1, surface=screen)
    pygame.display.update()

pygame.quit()