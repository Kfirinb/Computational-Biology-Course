"""
Computational Biology: Second Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
May 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

from sys import argv

import pygame
import numpy as np
import random

# Basic grid design, courtesy of Auctux: https://www.youtube.com/watch?v=d73z8U0iUYE

import pygame
import pygame.font
import numpy as np
import random

class Grid:
    def __init__(self, width, height, offset, screen):

        self.chuncks_flag = 0
        self.columns = 33
        self.rows = 33
        self.screen = screen
        self.scale = width / (self.columns)
        self.size = (self.rows, self.columns)
        self.grid_array = np.ndarray(shape=(self.size)).astype(int)
        self.offset = offset
        self.conditionsArray = []  # Filled @ Def SetConditions
        self.setConditions()
        self.solutionsCollection = []

    def random2d_array(self, mode):

        if mode == 1:
            matrixToModify = self.grid_array
        else:
            matrixToModify = np.ndarray(shape=(self.size)).astype(int)

        for x in range(self.rows):
            for y in range(self.columns):
                if((x < 8) and (y < 8)):
                    matrixToModify[x][y] = 3
                elif((x < 8) or (y < 8)):
                    matrixToModify[x][y] = 2
                elif mode != 3:
                    matrixToModify[x][y] = random.randint(0, 1)

        if mode == 1:
            self.grid_array = matrixToModify
            self.solutionsCollection.append(matrixToModify)
        else:
            return matrixToModify

    def gridUpdater(self,off_color, on_color, surface):
        for x in range(self.rows):
            for y in range(self.columns):

                y_pos = (y) * self.scale
                yScale = self.scale - self.offset

                x_pos = (x) * self.scale
                xScale = self.scale - self.offset

                if self.grid_array[x][y] == 3:  # Corner cell
                    pygame.draw.rect(surface, (0, 0, 0), [x_pos, y_pos, xScale, yScale])

                elif self.grid_array[x][y] == 2:  # Conditional cell
                    pygame.draw.rect(surface, off_color, [x_pos, y_pos, xScale, yScale])
                    font = pygame.font.SysFont(None, 20)
                    if(y < 8):  # Lines 0-24
                        try:
                            if(self.conditionsArray[x - 8][y] != 0):
                                img = font.render(str(self.conditionsArray[x - 8][y]), True, (0,0,0))
                                self.screen.blit(img,(x_pos, y_pos))
                            # If condition is 0, we omit it and leaving the cell empty.
                        except:  # Handles the event the customer didn't included a condition at all.
                            continue

                    else:  # Lines 25-49
                        try:
                            if(self.conditionsArray[y + 17][x] != 0):
                                img = font.render(str(self.conditionsArray[y + 17][x]), True, (0, 0, 0))
                                self.screen.blit(img, (x_pos, y_pos))
                            # If condition is 0, we omit it and leaving the cell empty.
                        except:
                            continue
                elif self.grid_array[x][y] == 1:
                    #rect = pygame.draw.rect(surface, on_color, [x_pos, y_pos, xScale, yScale])
                    pygame.draw.rect(surface, on_color, [x_pos, y_pos, xScale, yScale])
                #else:
                elif self.grid_array[x][y] == 0:
                    #rect = pygame.draw.rect(surface, off_color, [x_pos, y_pos, xScale, yScale])
                    pygame.draw.rect(surface, off_color, [x_pos, y_pos, xScale, yScale])

        #self.comparePopulation()  # Working on the general population.
        #self.grid_array = next
        if(len(self.solutionsCollection) == 1):  # Generating second random solution
            self.solutionGenerator(3, 100)
        else:
            mode = random.randint(1, 2)  # Determining if crossover is going to be on column or row
            pivot = random.randint(0, 24)  # Determining the the point in which we perform the cross over.
            self.solutionGenerator(mode, pivot)

    def setConditions(self):  # Reads the conditions from the input file.
        counter = 0
        file = open(argv[1], "r+")
        for line in file:
            if(counter == 50):
                break
            lineOfConditions = []
            for condition in line.split():
                lineOfConditions.append(int(condition))
            self.conditionsArray.append(lineOfConditions)
            counter += 1
        file.close()
        self.properSorter()

    def properSorter(self):
        for lineOfConditions in self.conditionsArray:
            properFlag = 0  # This flag checks for the sanity of the conditions array. It gives a green light to proceed
            # iff all 0's are before the non zero conditions.
            while properFlag == 0:
                somethingChangedFlag = 0
                for index in range(len(lineOfConditions)):
                    if index < len(lineOfConditions) - 1:
                        if lineOfConditions[index] != 0 and lineOfConditions[index + 1] == 0:
                            lineOfConditions[index + 1] = lineOfConditions[index]
                            lineOfConditions[index] = 0
                            somethingChangedFlag = 1
                if somethingChangedFlag == 0:
                    properFlag = 1

    def solutionGenerator(self, mode, pivot):
        solution1 = self.random2d_array(3)
        solution2 = self.random2d_array(3)

        if mode == 1: # Rows
            for x in range(8, pivot + 8):
                for y in range(8, self.rows):
                    solution1[x][y] = self.solutionsCollection[-2][x][y]
                    solution2[x][y] = self.solutionsCollection[-1][x][y]
            for x in range(pivot + 8, self.columns):
                for y in range(8, self.rows):
                    # if x == pivot + 8:
                      #  solution1[x][y] = 10
                       # solution1[x][y] = 10
                    #else:
                    solution1[x][y] = self.solutionsCollection[-1][x][y]
                    solution2[x][y] = self.solutionsCollection[-2][x][y]

            bestSolution = self.MatrixGrade(solution1, solution2)
            self.solutionsCollection.append(bestSolution)

        if mode == 2:  # Cols
            for x in range(8, self.columns):
                for y in range(8, pivot + 8):
                    solution1[x][y] = self.solutionsCollection[-2][x][y]
                    solution2[x][y] = self.solutionsCollection[-1][x][y]

            for x in range(8, self.columns):
                for y in range(pivot + 8, self.rows):
                    # if y == pivot + 8:
                      #  solution1[x][y] = 20
                       # solution1[x][y] = 20
                    #else:
                    solution1[x][y] = self.solutionsCollection[-1][x][y]
                    solution2[x][y] = self.solutionsCollection[-2][x][y]

            bestSolution = self.MatrixGrade(solution1, solution2)
            self.solutionsCollection.append(bestSolution)

        if mode == 3:
            solution2 = self.random2d_array(2)
            self.solutionsCollection.append(solution2)

        #print(len(self.solutionsCollection))
        self.solutionLoader()

    def solutionLoader(self):
        self.grid_array = self.solutionsCollection[-1]

    def comparePopulation(self, solution):
        # print(self.conditionsArray)
        counter = 0
        distanceAtColsFromDesiredPopulation = []
        distanceAtRowsFromDesiredPopulation = []

        for line in self.conditionsArray:
            desiredPopulation = 0
            for condition in line:
                desiredPopulation += condition
            #print(desiredPopulation)
            if counter < 25:  # Working on columns
                population = 0
                for y in range(8, self.rows):
                    population += solution[8 + counter][y]
                #print("Actual population of column " + str(counter) + " is " + str(population)
                     #+ " while the desired population is: " + str(desiredPopulation))
                distanceAtColsFromDesiredPopulation.append(population - desiredPopulation)
            else:  # Working on rows
                population = 0
                for x in range(8, self.columns):
                    population += solution[x][counter - 17]

                #print("Actual population of row " + str(counter - 25) + " is " + str(population)
                 #     + " while the desired population is: " + str(desiredPopulation))
                distanceAtRowsFromDesiredPopulation.append(population - desiredPopulation)
            counter += 1  # Proceeding to the next column/row.

        distancesToReturn = []
        distancesToReturn.append(distanceAtRowsFromDesiredPopulation)
        distancesToReturn.append(distanceAtColsFromDesiredPopulation)

        return distancesToReturn  # This array consists of the following members: [0] At Rows, [1] At Cols.

    # The following func transfers the population according to the requirements.
    def populationManager(self, distanceFromDesiredPopulation, mode):  # Mode 0 for cols, mode 1 for rows
        minimum = distanceFromDesiredPopulation[0]
        minimumIndex = 0
        maximum = distanceFromDesiredPopulation[0]
        maximumIndex = 0

        for index in range(len(distanceFromDesiredPopulation)):
            if distanceFromDesiredPopulation[index] < minimum and distanceFromDesiredPopulation[index] != 0:
                minimum = distanceFromDesiredPopulation[index]
                minimumIndex = index
            if distanceFromDesiredPopulation[index] > maximum and distanceFromDesiredPopulation[index] != 0:
                maximum = distanceFromDesiredPopulation[index]
                maximumIndex = index

        if minimum >= 0 or maximum <= 0:  # The black cells in all candidates exceeding or failing the desired
            return  # We cannot do anything in those cases.

        #print(minimumIndex)
        #print(minimum)
        self.populationMoverAboveBelow(minimumIndex, maximumIndex, mode)

    def populationMoverAboveBelow(self, minimumIndex, maximumIndex, mode):
        # print("Maximum: " + str(maximumIndex) + " Minimum: " + str(minimumIndex))

        next = np.ndarray(shape=(self.size)).astype(int)

        for x in range(self.rows):
            for y in range(self.columns):
                next[x][y] = self.grid_array[x][y]

        # print(next)

        if mode == 0:  # Working on cols
            for index in range(8, self.rows):
                if self.grid_array[maximumIndex][index] - self.grid_array[minimumIndex][index] > 0:  # The maximum cell is 1
                    # and the min in 0
                    next[maximumIndex + 8][index] = 0
                    next[minimumIndex + 8][index] = 1
                    break

        if mode == 1:  # Working on rows
            for index in range(8, self.columns):
                if self.grid_array[index][maximumIndex + 8] - self.grid_array[index][minimumIndex + 8] > 0:
                    next[index][maximumIndex + 8] = 0
                    next[index][minimumIndex + 8] = 1
                    break

        self.grid_array = next  # Here we affect the next grid

        # print(".......................")
        # print(counter)



    def chuncksCounter(self, solution):
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        w, h = 25, 2;
        chuncksArray = [[0 for x in range(w)] for y in range(h)]
        i = 0
        j = 0
        for row in range(8, self.rows):
            rows_chuncks = []
            rows_counter = 0
            for column in range(8, self.columns):
                if (solution[column][row] == 1):
                    rows_counter += 1
                elif (solution[column][row] == 0):
                    if (rows_counter > 0):
                        rows_chuncks.append(rows_counter)
                    rows_counter = 0
            if (rows_counter > 0):
                rows_chuncks.append(rows_counter)

            # if len(rows_chuncks) > 8:  todo: Do something!

            while(len(rows_chuncks) < 8):
                rows_chuncks.insert(0, 0)  # The first 0 reflects the position to which we insert the member - i.e the beginning of the array.
                # The second 0 represents the member we're inserting.
            chuncksArray[i][j] = rows_chuncks
            j += 1

            # print("Chuncks in row " + str(row-7) + " are: ")
            # print(rows_chunc
        i=1
        j=0
        for column in range(8, self.columns):
            cols_chuncks = []
            cols_counter = 0
            for row in range(8, self.rows):
                if (solution[column][row] == 1):
                    cols_counter += 1
                elif (solution[column][row] == 0):
                    if (cols_counter > 0):
                        cols_chuncks.append(cols_counter)
                    cols_counter = 0
            if (cols_counter > 0):
                cols_chuncks.append(cols_counter)

            # if len(cols_chuncks) > 8:  todo: Do something!

            while (len(cols_chuncks) < 8):
                cols_chuncks.insert(0, 0)
            chuncksArray[i][j] = cols_chuncks
            j += 1

            # print("Chuncks in column " + str(column-7) + " are: ")
            # print(cols_chuncks)
        return chuncksArray

    def compareLists(self,listConditions,listActual):
        #finalGrade=0.5
        #GradeReduction = 0.0625/2
        difference = 0
        IsOverlyLong = False
        if(len(listActual) > 8):  # If there are more than 8 chunks in the solution, the IsOverlyLong flag raise.
            #GradeReduction = 0.0625
            IsOverlyLong = True
        for i in range(8):  # Checks how many chunks are different between the requirements and the actual solution.
            if(listConditions[i] != listActual[i]):
                difference += 1
        return (difference,IsOverlyLong)

    def compareConditions(self, solution):
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        w, h = 25, 2;
        compareConditionsArray =[[0 for x in range(w)] for y in range(h)]
        i = 0
        chuncksArray = self.chuncksCounter(solution)
        for line in self.conditionsArray:
            if i < 25:  # Working on cols
                result = self.compareLists(line, chuncksArray[1][i])  # Returns a Tuple consisting of: [0] the number of chunks different
                # [1] A flag indicates if there are more than 8 chucks in a given row/col in the solution.
                if (result[1]):  # result[1] consists of the flag, (True or false). If Flag == True then...
                    compareConditionsArray[1][i] =  0.5 - (0.0625) * result[0]  # If the flag is raised the reduction is harsher.
                else:
                    compareConditionsArray[1][i] = 0.5 - (0.0625 / 2) * result[0]
                #compareConditionsArray[0][i] =  len(set(chuncksArray[0][i]) & set(line)) #checks the number of common elements in 2 lists
            else:  # Working on rows
                result = self.compareLists(line, chuncksArray[0][i - 25])
                if(result[1]):
                    compareConditionsArray[0][i-25] = 0.5 - (0.0625) * result[0]
                else:
                    compareConditionsArray[0][i - 25] = 0.5 - (0.0625/2) * result[0]

                #compareConditionsArray[1][i-25] = len(set(chuncksArray[0][i]) & set(line))
            i +=1

        return compareConditionsArray

    def GradeForConditions(self, solution):
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        # will be array for all grades for each column and row according to the conditions file order(meaning this array length will be 50)
        w, h = 25, 2
        gradesArray = [[0 for x in range(w)] for y in range(h)]
        #grade will be build by 50% comparePopulation and 50% chuncks comparasion
        comparePopulationArray = self.comparePopulation(solution) # consists population differences
        compareConditionsArray = self.compareConditions(solution) #consists grades already
        for i in range(0, 1):
            for j in range(0, 25):
                absoulte_difference = abs(comparePopulationArray[i][j])
                if absoulte_difference == 0:
                    gradesArray[i][j] = 0.5 + compareConditionsArray[i][j]
                else:
                    gradesArray[i][j] = 0.5 - 0.15*(absoulte_difference/10) + compareConditionsArray[i][j]
        for i in range(0, 1):

            '''for j in range(0, 25):
                if i == 0:
                    print("Grade for row ")
                else:
                    print("Grade for column ")
                print(str(j) + " is " + str(gradesArray[i][j]))'''

        return gradesArray

    def gradeCalculator(self, gradesArray):
        grade = 0
        for i in range(0, 1):
            for j in range(0, 25):
                grade += gradesArray[i][j]

        grade /= 50  # Calculating the average after going through the 2D array
        return grade  # A float

    def MatrixGrade(self, solution1, solution2):
        grade1 = self.gradeCalculator(self.GradeForConditions(solution1))  # Expecting to get a float to grade1
        print("Grade1: " + str(grade1))
        grade2 = self.gradeCalculator(self.GradeForConditions(solution2))  # Expecting to get a float to grade2
        print("Grade2: " + str(grade2))

        if(grade1 >= grade2):
            bestSolution = solution1
        else:
            bestSolution = solution2

        return bestSolution