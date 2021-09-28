"""
Computational Biology: First Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
March 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

import pygame
import numpy as np
import random

CHECK = True

min_vac_per = 100
# Basic Game of Life Mechanics Curtsy of Auctux https://www.youtube.com/watch?v=d73z8U0iUYE
class Grid:
    def __init__(self, width, height, scale, offset,Pv_max = 18, Pv_threshold = 4, Ph_max = 8, Ph_threshold = 0):
        self.scale = scale
        self.columns = 500
        self.rows = 500
        self.size = (self.rows, self.columns)
        self.grid_array = np.ndarray(shape = (self.size))
        self.offset = offset
        self.number_of_iterations = 0
        self.Pv_max = Pv_max
        self.Pv_threshold = Pv_threshold
        self.Ph_max = Ph_max
        self.Ph_threshold = Ph_threshold

    def random2d_arrayDist(self, sickPopulation, healthyPopulation, immunePopulation):
        for x in range(self.rows):
            for y in range(self.columns):
                self.grid_array[x][y] = -2

        for index in range(sickPopulation):
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            T = 10  # Factor of healing
            self.grid_array[x][y] = T  # 10

        for index in range(healthyPopulation):
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = 0

        for index in range(immunePopulation):
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = -1

    def random2d_arrayRandomPopulation(self, population):
        for x in range(self.rows):
            for y in range(self.columns):
                self.grid_array[x][y] = -2

        for index in range(population):
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            T = 10
            state = random.randint(-1, 1)
            if (state == 1):
                state = T
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = state



    def Conway(self, off_color, on_color, surface):  # Basic game of life mechanics, not in use in final submission.
        for x in range(self.rows):
            for y in range(self.columns):
                y_pos = y * self.scale
                x_pos = x * self.scale

                if self.grid_array[x][y] == 1:
                    pygame.draw.rect(surface, on_color, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                else:
                    pygame.draw.rect(surface, off_color, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
        next = np.ndarray(shape = (self.size))
        for x in range(self.rows):
            for y in range(self.columns):
                state = self.grid_array[x][y]
                neighbours = self.get_neighbours(x, y)
                if state == 0 and neighbours == 3:
                   next[x][y] = 1
                elif state == 1 and (neighbours < 2 or neighbours > 3):
                   next[x][y] = 0
                else:
                    next[x][y] = state
        self.grid_array = next

    def Covid(self, offColor, healthyColor, sickColor, immuneColor, surface):  # Terrifying function name (Our creation) :3
        global CHECK
        global min_vac_per
        for x in range(self.rows):
            for y in range(self.columns):
                y_pos = y * self.scale
                x_pos = x * self.scale

                if self.grid_array[x][y] >= 1:
                    pygame.draw.rect(surface, sickColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == 0:
                    pygame.draw.rect(surface, healthyColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == -1:
                    pygame.draw.rect(surface, immuneColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == -2:
                    pygame.draw.rect(surface, offColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])

        next = np.ndarray(shape=(self.size))
        for x in range(self.rows):
            for y in range(self.columns):
                state = self.grid_array[x][y]
                neighbours = self.get_neighbours(x, y)
                if state == -1:
                    #Pv_max = 18
                    #Pv_threshold = 4
                    immune_system = random.randint(int(self.Pv_threshold), int(self.Pv_max))
                    #immune_system += 1
                    #immune_system *= 2
                    if immune_system < neighbours:
                        next[x][y] = 10
                    else:
                        next[x][y] = -1

                if state == 0:
                    #Ph_max = 8
                    #Ph_threshold = 0
                    if(int(self.Ph_threshold) >= int(self.Ph_max)):
                        immune_system = self.Ph_threshold
                    else:
                        immune_system = random.randint(int(self.Ph_threshold), int(self.Ph_max))
                    if immune_system < neighbours:
                        next[x][y] = 10 #State 10 means the cell becomes sick for the next 10 generations
                    else:
                        next[x][y] = 0 #State 0 means the cell remained healthy

                elif state >= 1:
                    if state == 1:
                        next[x][y] = -1
                    else:
                        next[x][y] = state - 1
                        # next[x][y] = state

                elif state == -2:
                    next[x][y] = -2
                self.move(x, y, next)
        self.number_of_iterations += 1
        print("Number of generations passed: " + str(self.number_of_iterations)+"\n")
        vac_per,sick_per,total = self.printPopulationStatus(True)

        if(vac_per > sick_per and CHECK == True):
            min_vac_per = vac_per
            CHECK = False
        if(sick_per <= 1):
            return min_vac_per
        self.grid_array = next


    def move(self, x, y, next):  # Our in-use function to transport the population o the grid.
        n = random.randint(-1, 1)
        m = random.randint(-1, 1)

        x_edge = (x + n + self.rows) % self.rows
        y_edge = (y + m + self.columns) % self.columns
        if((self.grid_array[x_edge][y_edge] == -2) and (next[x_edge][y_edge] == -2)): #if the neighbor cell is empty and no one else saved it for the next round
            next[x_edge][y_edge] = self.grid_array[x][y]
            next[x][y] = -2

    def moveParanoid(self, x, y, next):  # A second tactic to help us transporting the population on the grid.
        for x in range(self.rows):
            for y in range(self.columns):
                next[x][y] = -2

        for x in range(self.rows):
            for y in range(self.columns):
                if (self.grid_array[x][y] == -2):
                    continue
                n = random.randint(-1, 1)
                m = random.randint(-1, 1)
                futureX = (x + n) % self.rows
                futureY = (y + m) % self.columns
                while(next[futureX][futureY] != -2):
                    n = random.randint(-1, 1)
                    m = random.randint(-1, 1)
                    futureX = (x + n) % self.rows
                    futureY = (y + m) % self.columns
                next[futureX][futureY] = self.grid_array[x][y]



    def get_neighbours(self, x, y):
        total = 0
        for n in range(-1, 2):
            for m in range(-1, 2):
                x_edge = (x + n + self.rows) % self.rows
                y_edge = (y + m + self.columns) % self.columns
                if(self.grid_array[x_edge][y_edge] >= 1):
                    # total += self.grid_array[x_edge][y_edge]
                    total += 1

        if(self.grid_array[x][y] >= 1):
            total -= 1
        return total

    def printPopulationStatus(self,investigation= False):
        sick_counter = 0
        vaccinated_counter = 0
        total_population = 0
        for x in range(self.rows):
            for y in range(self.columns):
                state = self.grid_array[x][y]
                if(state == -1):
                    vaccinated_counter += 1
                    total_population += 1
                elif(state >= 1):
                    sick_counter += 1
                    total_population += 1
                elif(state == 0):
                    total_population += 1
        vaccinated_percentage = (vaccinated_counter / total_population) * 100
        sick_percentage = (sick_counter / total_population) * 100
        if(investigation):
            print("After " + str(self.number_of_iterations) + " generations, we have: " +
                  str(sick_percentage) + "% sick people and " + str(
                vaccinated_percentage) +
                  "% vaccinated among a population of: " + str(total_population) + "people")
            return vaccinated_percentage,sick_percentage,total_population


        return 0,0,0
