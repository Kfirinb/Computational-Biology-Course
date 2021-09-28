"""
Computational Biology: First Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
March 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

import pygame
import numpy as np
import random

# Basic Game of Life Mechanics Curtsy of Auctux https://www.youtube.com/watch?v=d73z8U0iUYE

#  The following class is the heart of our submission. It crafts the 500 X 500 grid, and run the game with the desired options.


class Grid:
    def __init__(self, width, height, scale, offset, T, phTresh, phMax, pvTresh, pvMax):
        self.scale = scale
        self.columns = 500
        self.rows = 500
        self.size = (self.rows, self.columns)
        self.grid_array = np.ndarray(shape = (self.size))
        self.offset = offset
        self.number_of_iterations = 0
        self.T = T
        self.phTresh = phTresh
        self.phMax = phMax
        self.pvTresh = pvTresh
        self.pvMax = pvMax

    #  The function that supports the argument "d", it receives three values reflecting the sizes of the three populations.
    #  Each person among the populations is being placed randomly on the grid.
    def random2d_arrayDist(self, sickPopulation, healthyPopulation, immunePopulation):
        for x in range(self.rows):
            for y in range(self.columns):  # Fills the entire grid with empty cells, in preparation to the next stage.
                self.grid_array[x][y] = -2

        for index in range(sickPopulation):  # Places the sick population on the grid
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):  # Ensures an empty cell was selected for placing.
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = self.T  # 10

        for index in range(healthyPopulation):  # Places the healthy population on the grid
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = 0

        for index in range(immunePopulation):  # Places the immune population on the grid
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)
            while (self.grid_array[x][y] != -2):
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = -1

    #  This function provides support for the argument "r". It requires the general size of the entire population (Ns + Nv + Nh)
    #  And randomly decides the condition of each individual and their location on the grid in a random fashion.
    def random2d_arrayRandomPopulation(self, population):
        for x in range(self.rows):
            for y in range(self.columns):
                self.grid_array[x][y] = -2

        for index in range(population):
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.columns - 1)

            state = random.randint(-1, 1)  # Uniformly randomize the condition of the person. -1 for immune, 0 for healthy, 1 for sick.
            if (state == 1):
                state = self.T  # Applies number of generations for sick person to remain sick.
            while (self.grid_array[x][y] != -2):  # Ensures an empty cell was selected for placing.
                x = random.randint(0, self.rows - 1)
                y = random.randint(0, self.columns - 1)
            self.grid_array[x][y] = state  # After the cell selected, and condition determined, we are placing the person on the grid.

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

    def Covid(self, offColor, healthyColor, sickColor, immuneColor, surface, enableMove):  # Terrifying function name (Our creation) :3
        for x in range(self.rows):
            for y in range(self.columns):
                y_pos = y * self.scale
                x_pos = x * self.scale

            # The following conditions, paint the cells with their corresponding colors.
                if self.grid_array[x][y] >= 1:
                    pygame.draw.rect(surface, sickColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == 0:
                    pygame.draw.rect(surface, healthyColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == -1:
                    pygame.draw.rect(surface, immuneColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                elif self.grid_array[x][y] == -2:
                    pygame.draw.rect(surface, offColor, [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])

        next = np.ndarray(shape=(self.size))  #  Prepares a NumPy array for next iteration.
        for x in range(self.rows):
            for y in range(self.columns):
                state = self.grid_array[x][y]
                neighbours = self.get_neighbours(x, y)
                if state == -1:  # Deciding the destiny of an IMMUNE person.
                    immune_system = random.randint(self.pvTresh, self.pvMax)  # Randomizes immunity strength uniformly. in the desired range

                    if immune_system < neighbours:  # Immune person becomes sick.
                        next[x][y] = self.T
                    else:  # Immunity system strength was efficient enough. Hence person remains immune.
                        next[x][y] = -1

                if state == 0:  # Deciding the destiny of a HEALTHY person.
                    immune_system = random.randint(self.phTresh, self.phMax)  # Randomizes immunity strength uniformly.
                    if immune_system < neighbours:  # Healthy person becomes sick.
                        next[x][y] = self.T
                    else:  # Immunity system strength was efficient enough. Hence person remains healthy.
                        next[x][y] = 0

                elif state >= 1:  # Dealing with sick person.
                    if state == 1:  # Sick person was sick for T generations.
                        next[x][y] = -1
                    else:  # Sick person remains sick.
                        next[x][y] = state - 1  # Deduction one generation from the sick person.

                elif state == -2:  # Dealing with an empty cell
                    next[x][y] = -2

                if enableMove:  # Handling the final argument
                    self.move(x, y, next)  # Calling the dedicated move function if customer opted for this option.

        self.number_of_iterations += 1  # Reporting data to console.
        print("Number of generations passed: " + str(self.number_of_iterations)+"\n")
        self.countPopulation()
        self.grid_array = next  # Loading next grid arrangement.

    #  Supporting "russian" argument. Provides an alternative to Covid function.
    def russianRoulette(self, offColor, healthyColor, sickColor, immuneColor, surface, enableMove):
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
                if (state == 0):  # Healthy but not immune
                    next[x][y] = 0
                    for _ in range(neighbours):
                        result = random.randint(0, 9)
                        if (result == 0):
                            next[x][y] = self.T
                            break  # The cell lost the game and will turn sick.

                if (state == -1):  # Immune
                    next[x][y] = -1
                    for _ in range(neighbours):
                        result = random.randint(0, 99)
                        if (result == 0):
                            next[x][y] = 10
                            break  # The cell lost the game and will turn sick.

                elif state >= 1:
                    if state == 1:
                        next[x][y] = -1
                    else:
                        next[x][y] = state - 1
                        # next[x][y] = state

                elif state == -2:
                    next[x][y] = -2
                if(enableMove):
                    self.move(x, y, next)

        self.number_of_iterations += 1
        print("Number of generations passed: " + str(self.number_of_iterations) + "\n")
        self.countPopulation()
        self.grid_array = next

    def move(self, x, y, next):  # Our in-use function to transport the population on the grid.
        n = random.randint(-1, 1)  # Randomizes an offset for the move uniformly
        m = random.randint(-1, 1)  # Randomizes an offset for the move uniformly

        x_edge = (x + n + self.rows) % self.rows
        y_edge = (y + m + self.columns) % self.columns
        if((self.grid_array[x_edge][y_edge] == -2) and (next[x_edge][y_edge] == -2)):  # if the neighbor cell is empty and no one else saved it for the next round
            next[x_edge][y_edge] = self.grid_array[x][y]
            next[x][y] = -2

    def moveParanoid(self, x, y, next):  # A second tactic to help us transporting the population on the grid. Wasn't used
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
                while(next[futureX][futureY] != -2):  # Forcing a move of some sort, unless it was RANDOMLY decided to stay in place.
                    n = random.randint(-1, 1)
                    m = random.randint(-1, 1)
                    futureX = (x + n) % self.rows
                    futureY = (y + m) % self.columns
                next[futureX][futureY] = self.grid_array[x][y]

    def get_neighbours(self, x, y):  # Classifying the neighbors of the cell.
        total = 0
        for n in range(-1, 2):
            for m in range(-1, 2):
                x_edge = (x + n + self.rows) % self.rows
                y_edge = (y + m + self.columns) % self.columns
                if(self.grid_array[x_edge][y_edge] >= 1):
                    # total += self.grid_array[x_edge][y_edge]
                    total += 1  # Counting the number of sick neighbors.

        if(self.grid_array[x][y] >= 1):  # Deducting the cell itself if necessary.
            total -= 1
        return total

    def countPopulation(self):
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
        #print("Percentage of sick citizens:" + (sick_counter/total_population)*100 +"\n")
        print("After " + str(self.number_of_iterations) + " generations, we have: " +
              str((sick_counter/total_population)*100) + "% sick people and " + str((vaccinated_counter/total_population)*100) +
              "% vaccinated among a population of: " + str(total_population) + " people")
