"""
Computational Biology: First Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
March 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

import pygame
import sys
import os
import grid

# Basic Game of Life Mechanics Curtsy of Auctux https://www.youtube.com/watch?v=d73z8U0iUYE

os.environ["SDL_VIDEO_CENTERED"] = '1'
width = 900 #1440
height = 500 #720
size = (width, height)

pygame.init()
pygame.display.set_caption("Computational Biology - Assignment 1 By CUCUMBER and Kfir Inbal")
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
fps = 60

GREEN = (38, 188, 27)  # RGB values of green.
YELLOW = (255, 255, 0)  # RGB values of yellow.
RED = (255, 0, 0)  # RGB values of red.
WHITE = (255, 255, 255)  # RGB values of white.
BLACK = (0, 0, 0)  # RGB values of black.

scaler = 5
offset = 0.1

argLen = len(sys.argv)  # Saving the number of provided arguments. We need either 5 or 7. (4 or 6 args from the customer).

if(argLen == 10):
    Grid = grid.Grid(width, height, scaler, offset,
                     T=int(sys.argv[5]), phTresh=int(sys.argv[6]), phMax=int(sys.argv[7]),
                     pvTresh=int(sys.argv[8]), pvMax=int(sys.argv[9]))  # Creating the grid. Along with the parameters

elif(argLen == 12):
    Grid = grid.Grid(width, height, scaler, offset,
                     T=int(sys.argv[7]), phTresh=int(sys.argv[8]), phMax=int(sys.argv[9]),
                     pvTresh=int(sys.argv[10]), pvMax=int(sys.argv[11]))  # Creating the grid. Along with the parameters

else:
    pygame.quit()

# nameOfApp r 50000 covid true 10 0 8 4 18
# nameOfApp d 20000 10000 20000 russian false 10 0 8 4 18



if(argLen == 10 or argLen == 12):
    if(sys.argv[1] == "r"):  # r stands for "Random", if this argument provided, the customer just needs to provide the size of population, and the app does the rest.
        Grid.random2d_arrayRandomPopulation(population=int(sys.argv[2]))  # Required size received in the following argument
    elif(sys.argv[1] == "d"):  # d stands for "Distribution", if this argument provided, the customer needs to specify the size of Ns, Nh and Nv.
        Grid.random2d_arrayDist(sickPopulation=int(sys.argv[2]), healthyPopulation=int(sys.argv[3]), immunePopulation=int(sys.argv[4])) # Required sizes received in the following three arguments

run = True
while run:
    clock.tick(fps)
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    if(argLen == 10 and sys.argv[1] == "r"):
        if(sys.argv[3] == "covid"):  # Second from last argument specifies the mode desired: "covid" for the classic, researched mode and
            # "russian" for the Russian Roulette mode.
            Grid.Covid(offColor=BLACK, healthyColor=YELLOW, sickColor=RED, immuneColor=GREEN, surface=screen,
                       enableMove=sys.argv[4])  # The option of movement is provided in the final argument.
        elif(sys.argv[3] == "russian"):
            Grid.russianRoulette(offColor=BLACK, healthyColor=YELLOW, sickColor=RED, immuneColor=GREEN, surface=screen,
                                 enableMove=sys.argv[4])  # The option of movement is provided in the final argument.

    elif(argLen == 12 and sys.argv[1] == "d"):
        if (sys.argv[5] == "covid"):
            Grid.Covid(offColor=BLACK, healthyColor=YELLOW, sickColor=RED, immuneColor=GREEN, surface=screen,
                       enableMove=sys.argv[6])
        elif (sys.argv[5] == "russian"):
            Grid.russianRoulette(offColor=BLACK, healthyColor=YELLOW, sickColor=RED, immuneColor=GREEN, surface=screen,
                                 enableMove=sys.argv[6])

    else:  # Something is wrong with the provided arguments.
        pygame.quit()

    pygame.display.update()
pygame.quit()
