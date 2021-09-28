"""
Computational Biology: First Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
March 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

import pygame
import sys
import os
import gridInvestigateP as grid
import matplotlib.pyplot as plt
import math
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

GREEN = (38, 188, 27)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

scaler = 5
offset = 0.1
Investigation = True
Ph_array = []
Ph_result_array = []

def createGrid(width, height, scaler, offset,Pv_max = 18, Pv_threshold = 4, Ph_max = 8, Ph_threshold = 0):
    Grid = grid.Grid(width, height, scaler, offset,Pv_max, Pv_threshold, Ph_max, Ph_threshold)
    Grid.random2d_arrayDist(sickPopulation=80000, healthyPopulation=80000, immunePopulation=80000)
    return Grid

def calculatePhMax(Ph_threshold,Ph):
    sum_of_neighbors =0
    for neighbors in range(0,8):
        if(Ph_threshold < neighbors):
            sum_of_neighbors += neighbors

    Ph_max = abs(((sum_of_neighbors-Ph_threshold-9*Ph*Ph_threshold-9*Ph)/(9*Ph)))
    return Ph_max
   # if(Ph_max <= Ph_threshold):
       # temp = Ph_max
       # Ph_max = Ph_threshold
       # Ph_threshold = temp


def investigateP():
    #sum_of_neighbors = 0+1+2+3+4+5+6+7+8
    for Ph_threshold in range(0,8):
        for index in range(1,10,2):
            Ph = float(index/10) # for Ph in range(0.1,1,0.2)
            Ph_array.append(Ph)
            Ph_max = calculatePhMax(Ph_threshold,Ph)
            #Ph_max = ((sum_of_neighbors-Ph_threshold-9*Ph*Ph_threshold-9*Ph)/(9*Ph)) #sum_of_neighbors=36
            #if(int(Ph_max) == Ph_threshold):
                #Ph_max += 1
            result=run_game(createGrid(width, height, scaler, offset, 18,4,Ph_max, Ph_threshold),True,True)
            Ph_result_array.append(result)
    #print("RESULT=", result)
    print("Ph array:",Ph_array)
    print("Ph results array:",Ph_result_array)
    plt.plot(Ph_array, Ph_result_array,label="Minimum vaccinated population percentage by Ph")
    plt.axis([0, 1, 0, 100])
    plt.show()


# Grid.random2d_arrayRandomPopulation(population=240000)
#numberOfIterations = 0
#print("Number of generations passed: " + str(numberOfIterations))
def run_game(Grid,run = True, investigate = False):
    run = run
    result = None
    while run:
        clock.tick(fps)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if(investigate):
            result = Grid.Covid(offColor = BLACK, healthyColor = YELLOW, sickColor = RED, immuneColor = GREEN, surface = screen)
            print(result)
            #run = False
        else:
            Grid.Covid(offColor = BLACK, healthyColor = YELLOW, sickColor = RED, immuneColor = GREEN, surface = screen)

        pygame.display.update()
        if result != None:
            run = False
            return result
    pygame.quit()

#run_game(createGrid(width, height, scaler, offset,Pv_max = 18, Pv_threshold = 4, Ph_max = 8, Ph_threshold = 0))
investigateP()
