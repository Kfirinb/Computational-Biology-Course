"""
Computational Biology: Second Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
May 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""

from sys import argv

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
        self.totalDesiredPopulation = 0
        self.setConditions()
        self.solutionsNextGeneration = []
        self.solutionsCurrentGeneration = []
        self.currentGrade = 0  # Represents the grade of the best solution
        self.stuckRate = 0  # Determines for how many iterations the currentGrade field didn't changed.
        self.bestSolution = 0
        self.numberOfSolutionsToHandle = 10
        self.expanderRate = 0
        self.mode = int(argv[2])
        self.iteration_num = 0

    def setConditions(self):  # Reads the conditions from the input file.
        counter = 0
        file = open(argv[1], "r+")
        for line in file:
            if (counter == 50):
                break
            lineOfConditions = []
            for condition in line.split():
                lineOfConditions.append(int(condition))
                self.totalDesiredPopulation += int(condition)
            while len(lineOfConditions) < 8:
                lineOfConditions.append(int(0))
            self.conditionsArray.append(lineOfConditions)
            counter += 1
        file.close()
        self.totalDesiredPopulation = int(self.totalDesiredPopulation / 2)
        self.properSorter()

    def properSorter(self):  # Correctly format the conditions (0's before the rest).
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

    def templateConstructor(self):  # Creates a basic pattern of solution consisting the conditions.
        matrixToModify = np.ndarray(shape=(self.size)).astype(int)
        for x in range(self.rows):
            for y in range(self.columns):
                if ((x < 8) and (y < 8)):
                    matrixToModify[x][y] = 3
                elif ((x < 8) or (y < 8)):
                    matrixToModify[x][y] = 2
                else:
                    matrixToModify[x][y] = 0

        return matrixToModify

    def random2d_array(self, numberOfSolutions=None):  # Fills the sub-matrix of 25 X 25 randomly
        # fulfilling the columns requirements
        if (numberOfSolutions == None):
            numberOfSolutions = self.numberOfSolutionsToHandle
        for _ in range(numberOfSolutions):
            matrixToModify = self.templateConstructor()

            col = 0
            row = 8
            for line in self.conditionsArray:
                conditionsSum = sum(line)
                numberOfConditions = len([i for i in line if (i > 0)])
                numberOfFreeIndexes = 25 - (conditionsSum + (numberOfConditions - 1))
                indexStarter = random.randint(0, numberOfFreeIndexes)
                if col < 25:
                    for i in range(8):
                        if (line[i] > 0):
                            for j in range(line[i]):
                                matrixToModify[row + indexStarter][col + 8] = 1
                                row += 1
                        else:
                            continue
                        conditionsSum = conditionsSum - line[i]
                        numberOfConditions = numberOfConditions - 1
                        num = 32 - (row + indexStarter) - ((conditionsSum) + (numberOfConditions)) + 2
                        randnum = random.randint(1, num)
                        for r in range(randnum):
                            row += 1
                else:
                    break

                col += 1
                row = 8

            if self.mode == 2:  # Working on LeMark
                if len(self.solutionsCurrentGeneration) > 1:
                    bestCurrentSolution = np.copy(self.solutionsCurrentGeneration[-1][0])
                    gradeMatrixBefore = self.matrixGrade(matrixToModify)
                    gradeCurrentBefore = self.matrixGrade(bestCurrentSolution)
                    copymatrixToModify, copybestCurrentSolution = self.optimizer(np.copy(matrixToModify),
                                                                                 np.copy(bestCurrentSolution))
                    if (self.matrixGrade(copymatrixToModify) > gradeMatrixBefore):
                        matrixToModify = copymatrixToModify
                    if (self.matrixGrade(copybestCurrentSolution) > gradeCurrentBefore):
                        bestCurrentSolution = copybestCurrentSolution
                        self.solutionsCurrentGeneration[-1][0] = bestCurrentSolution
                        self.solutionsCurrentGeneration[-1][1] = self.matrixGrade(bestCurrentSolution)

            solution = [matrixToModify, 0]
            solution[1] = self.matrixGrade(solution[0])

            if self.mode == 1:  # Working on Darwin
                if len(self.solutionsCurrentGeneration) > 1:
                    bestCurrentSolution = np.copy(self.solutionsCurrentGeneration[-1][0])
                    gradeMatrixBefore = self.matrixGrade(matrixToModify)
                    gradeCurrentBefore = self.matrixGrade(bestCurrentSolution)
                    copymatrixToModify, copybestCurrentSolution = self.optimizer(np.copy(matrixToModify),
                                                                                 np.copy(bestCurrentSolution))
                    if (self.matrixGrade(copymatrixToModify) > gradeMatrixBefore):
                        matrixToModify = copymatrixToModify
                    if (self.matrixGrade(copybestCurrentSolution) > gradeCurrentBefore):
                        bestCurrentSolution = copybestCurrentSolution
                        self.solutionsCurrentGeneration[-1][0] = bestCurrentSolution
                        # self.solutionsCurrentGeneration[-1][1] = self.matrixGrade(bestCurrentSolution)
                    solution[0] = matrixToModify

            self.solutionsCurrentGeneration.append(solution)

    def specificColInitializer(self, numberOfCol, solution):  # Executes the mutation mechanic.
        for row in range(8, 33):
            solution[row][numberOfCol] = 0
        row = 8
        line = self.conditionsArray[numberOfCol - 8]
        conditionsSum = sum(line)
        numberOfConditions = len([i for i in line if (i > 0)])
        numberOfFreeIndexes = 25 - (conditionsSum + (numberOfConditions - 1))
        indexStarter = random.randint(0, numberOfFreeIndexes)
        for i in range(8):
            if (line[i] > 0):
                for j in range(line[i]):
                    solution[row + indexStarter][numberOfCol] = 1
                    row += 1
            else:
                continue
            conditionsSum = conditionsSum - line[i]
            numberOfConditions = numberOfConditions - 1
            num = 32 - (row + indexStarter) - ((conditionsSum) + (numberOfConditions)) + 2
            randnum = random.randint(1, num)
            for r in range(randnum):
                row += 1

    def gridUpdater(self, off_color, on_color, surface):  # Updates the screen with the next solution.

        if (self.stuckRate >= 10):
            self.solutionsCurrentGeneration = sorted(self.solutionsCurrentGeneration, key=lambda tup: tup[1])
            self.stuckSolver()
            self.stuckRate = 0

        self.solutionsCurrentGeneration = sorted(self.solutionsCurrentGeneration, key=lambda tup: tup[1])
        self.grid_array = self.solutionsCurrentGeneration[-1][0]
        self.bestSolution = self.solutionsCurrentGeneration[-1]
        bestCurrentGrade = self.matrixGrade(self.grid_array)
        if (bestCurrentGrade > 95 or self.iteration_num >= 5000):
            pauser = input("PAUSED")
        # print([self.solutionsCurrentGeneration[i][1] for i in range(10)])
        self.iteration_num += 1
        print("Iteration " + str(self.iteration_num) + ", best solution score: " + str(bestCurrentGrade))
        if (bestCurrentGrade - self.currentGrade) == 0:
            self.stuckRate += 1
            self.expanderRate += 1
        else:
            self.stuckRate = 0
            self.expanderRate = 0
            # print(self.stuckRate)
            self.currentGrade = bestCurrentGrade

        for x in range(self.rows):
            for y in range(self.columns):
                y_pos = (y) * self.scale
                yScale = self.scale - self.offset
                x_pos = (x) * self.scale
                xScale = self.scale - self.offset

                if self.grid_array[x][y] == 3:  # Corner cell
                    pygame.draw.rect(surface, (0, 0, 0), [y_pos, x_pos, xScale, yScale])

                elif self.grid_array[x][y] == 2:  # Conditional cell
                    pygame.draw.rect(surface, off_color, [x_pos, y_pos, xScale, yScale])
                    font = pygame.font.SysFont(None, 20)
                    if (y < 8):  # Lines 0-24
                        try:
                            if (self.conditionsArray[x - 8][y] != 0):
                                img = font.render(str(self.conditionsArray[x - 8][y]), True, (0, 0, 0))
                                self.screen.blit(img, (x_pos, y_pos))
                            # If condition is 0, we omit it and leaving the cell empty.
                        except:  # Handles the event the customer didn't included a condition at all.
                            continue

                    else:  # Lines 25-49
                        try:
                            if (self.conditionsArray[y + 17][x] != 0):
                                img = font.render(str(self.conditionsArray[y + 17][x]), True, (0, 0, 0))
                                self.screen.blit(img, (x_pos, y_pos))
                            # If condition is 0, we omit it and leaving the cell empty.
                        except:
                            continue
                elif self.grid_array[x][y] == 1:
                    pygame.draw.rect(surface, on_color, [y_pos, x_pos, xScale, yScale])
                elif self.grid_array[x][y] == 0:
                    pygame.draw.rect(surface, off_color, [y_pos, x_pos, xScale, yScale])

        self.solutionComposer()
        self.solutionsCurrentGeneration = self.solutionsNextGeneration
        self.solutionsNextGeneration = []

    #################COMPOSING SOLUTION####################################
    def solutionComposer(self):  # Creates the next collection of solutions.
        replicationMutationIndexes = []
        firstCrossOverFlag = 1
        new_solution = self.mutation(np.copy(self.solutionsCurrentGeneration[-1][0]))  # Mutation on best
        self.solutionsNextGeneration.append([new_solution, self.matrixGrade(new_solution)])
        self.solutionsNextGeneration.append(self.solutionsCurrentGeneration[-1])
        replicationMutationIndexes.append(self.numberOfSolutionsToHandle - 1)

        if (self.stuckRate > 0):
            Pmutation = 25 + self.stuckRate * 5
            if (Pmutation > 100):
                Pmutation = 100
        else:
            Pmutation = 25
        for index in range(1, self.numberOfSolutionsToHandle - 7 - (self.numberOfSolutionsToHandle - 10)):
            destiny = random.randint(0, 100)
            if destiny < Pmutation:  # Mutation on the solution
                new_solution = self.mutation(np.copy(self.solutionsCurrentGeneration[-1 - index][0]))
                self.solutionsNextGeneration.append([new_solution, self.matrixGrade(new_solution)])
                replicationMutationIndexes.append(self.numberOfSolutionsToHandle - index)
            else:  # replication on the solution
                self.solutionsNextGeneration.append(self.solutionsCurrentGeneration[-1 - index])
                replicationMutationIndexes.append(self.numberOfSolutionsToHandle - index)

        crossOverSkipFlag = 0

        for index in range(self.numberOfSolutionsToHandle - 4):
            if crossOverSkipFlag == 1:
                crossOverSkipFlag = 0
                continue
            if (self.stuckRate > 0):
                Pmutation = 50 + self.stuckRate * 3
                if Pmutation > 95:
                    Pmutation = 95
            else:
                Pmutation = 50
            if index == self.numberOfSolutionsToHandle - 5:
                PforIndex5 = Pmutation - 1

                thingToDo = random.randint(1, PforIndex5)  # 1% replication, 99% mutation, unable to do crossover
            else:
                thingToDo = random.randint(1, 100)  # 1% replication, 50% mutation, 49% crossover

            if thingToDo < Pmutation:  # Preparing a winner
                winner = random.randint(0, self.numberOfSolutionsToHandle - 4)
                while winner in replicationMutationIndexes:
                    winner = random.randint(0, self.numberOfSolutionsToHandle - 4)

            if thingToDo == 1:  # replication
                self.solutionsNextGeneration.append(self.solutionsCurrentGeneration[winner])
                replicationMutationIndexes.append(winner)

            elif thingToDo > 1 and thingToDo < Pmutation:  # Mutation
                new_solution = self.mutation(np.copy(self.solutionsCurrentGeneration[winner][0]))
                self.solutionsNextGeneration.append([new_solution, self.matrixGrade(new_solution)])
                replicationMutationIndexes.append(winner)

            else:  # Crossover:
                self.crossOverPrepare(firstCrossOverFlag)
                firstCrossOverFlag = 0
                crossOverSkipFlag = 1

    def crossOverPrepare(self, flag):  # Preparing a cross over.
        mode = random.randint(1, 2)  # Determining if crossover is going to be on column or row
        pivot = random.randint(8, 32)  # Determining the the point in which we perform the cross over.
        if flag == 1:
            index1 = self.numberOfSolutionsToHandle - 1
            index2 = self.numberOfSolutionsToHandle - 2
        else:
            index1 = random.randint(0, self.numberOfSolutionsToHandle - 1)
            index2 = random.randint(0, self.numberOfSolutionsToHandle - 1)
            while index2 == index1:
                index2 = random.randint(0, self.numberOfSolutionsToHandle - 1)

        self.crossOver(mode, pivot, self.solutionsCurrentGeneration[index1][0],
                       self.solutionsCurrentGeneration[index2][0])

    def crossOver(self, mode, pivot, firstParty, secondParty):  # Generates a cross over solution.
        part1 = firstParty[:, 0:pivot]
        part2 = secondParty[:, pivot:33]
        solution1 = np.concatenate((part1, part2), axis=1)
        part1 = secondParty[:, 0:pivot]
        part2 = firstParty[:, pivot:33]
        solution2 = np.concatenate((part1, part2), axis=1)
        tuple1 = [solution1, 0]
        tuple1[1] = self.matrixGrade(solution1)
        tuple2 = [solution2, 0]
        tuple2[1] = self.matrixGrade(solution2)
        self.solutionsNextGeneration.append(tuple1)
        self.solutionsNextGeneration.append(tuple2)

    def mutation(self, solution):  # Prepares a mutation.
        col = random.randint(8, 32)
        self.specificColInitializer(col, solution)
        return solution

    def stuckSolver(self):  # Resolves an early covariance situation
        if (self.expanderRate >= 100):
            firstBest = self.solutionsCurrentGeneration[self.numberOfSolutionsToHandle - 1]
            firstBestGrade = self.matrixGrade(firstBest[0])
            secondBest = firstBest
            # flagCanOptimize = 1

            for i in range(self.numberOfSolutionsToHandle - 2, -1, -1):
                secondBest = self.solutionsCurrentGeneration[i]
                if (firstBestGrade - self.matrixGrade(secondBest[0]) > 2):
                    break

            self.numberOfSolutionsToHandle += 5
            self.expanderRate = 0

        else:
            firstBest = self.solutionsCurrentGeneration[self.numberOfSolutionsToHandle - 1]
            secondBest = self.solutionsCurrentGeneration[self.numberOfSolutionsToHandle - 2]

        self.solutionsCurrentGeneration = []
        self.random2d_array(self.numberOfSolutionsToHandle - 2)
        self.solutionsCurrentGeneration.append(secondBest)
        self.solutionsCurrentGeneration.append(firstBest)

    ###################################################################################################################
    def comparePopulation(self, solution):  # Comparing the size of population in rows to the required size of population.
        # print(self.conditionsArray)
        counter = 0
        distanceAtColsFromDesiredPopulation = []
        distanceAtRowsFromDesiredPopulation = []

        for line in self.conditionsArray:
            desiredPopulation = 0
            for condition in line:
                desiredPopulation += condition
            # print(desiredPopulation)
            if counter < 25:  # Working on columns
                population = 0
                for y in range(8, self.rows):
                    population += solution[y][8 + counter]
                # print("Actual population of column " + str(counter) + " is " + str(population)
                # + " while the desired population is: " + str(desiredPopulation))
                distanceAtColsFromDesiredPopulation.append(population - desiredPopulation)
            else:  # Working on rows
                population = 0
                for x in range(8, self.columns):
                    population += solution[counter - 17][x]

                # print("Actual population of row " + str(counter - 25) + " is " + str(population)
                #     + " while the desired population is: " + str(desiredPopulation))
                distanceAtRowsFromDesiredPopulation.append(population - desiredPopulation)
            counter += 1  # Proceeding to the next column/row.

        distancesToReturn = []
        distancesToReturn.append(distanceAtRowsFromDesiredPopulation)
        distancesToReturn.append(distanceAtColsFromDesiredPopulation)

        return distancesToReturn  # This array consists of the following members: [0] At Rows, [1] At Cols.

    def chuncksCounter(self, solution):  # Counting the fragments (Chuncks) in each line.
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        w, h = 25, 2;
        chuncksArray = [[0 for x in range(w)] for y in range(h)]
        ###CHUNCKS IN ROWS###
        i = 0
        j = 0
        for row in range(8, self.rows):
            rows_chuncks = []
            rows_counter = 0
            for column in range(8, self.columns):
                if (solution[row][column] == 1):
                    rows_counter += 1
                elif (solution[row][column] == 0):
                    if (rows_counter > 0):
                        rows_chuncks.append(rows_counter)
                    rows_counter = 0
            if (rows_counter > 0):
                rows_chuncks.append(rows_counter)
            while (len(rows_chuncks) < 8):
                rows_chuncks.insert(0, 0)  # The first 0 reflects the position to which we insert the member
                # - i.e the beginning of the array. The second 0 represents the member we're inserting.
            chuncksArray[i][j] = rows_chuncks
            j += 1
        ###CHUNCKS IN COLUMNS###
        i = 1
        j = 0
        for column in range(8, self.columns):
            cols_chuncks = []
            cols_counter = 0
            for row in range(8, self.rows):
                if (solution[row][column] == 1):
                    cols_counter += 1
                elif (solution[row][column] == 0):
                    if (cols_counter > 0):
                        cols_chuncks.append(cols_counter)
                    cols_counter = 0
            if (cols_counter > 0):
                cols_chuncks.append(cols_counter)
            while (len(cols_chuncks) < 8):
                cols_chuncks.insert(0, 0)
            chuncksArray[i][j] = cols_chuncks
            j += 1
        return chuncksArray

    def compareLists(self, listConditions, listActual):  # Given a proposed solution and requirements, we compare them.
        difference = 0
        IsOverlyLong = False
        if (len(listActual) > 8):  # If there are more than 8 chunks in the solution, the IsOverlyLong flag raise.
            # GradeReduction = 0.0625
            IsOverlyLong = True
        for i in range(8):  # Checks how many chunks are different between the requirements and the actual solution.
            if (listConditions[i] != listActual[i]):
                difference += 1
        return (difference, IsOverlyLong)

    def compareConditions(self, solution):  # Compares the conditions to the proposed solution
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        w, h = 25, 2;
        compareConditionsArray = [[0 for x in range(w)] for y in range(h)]
        i = 0
        chuncksArray = self.chuncksCounter(solution)
        for line in self.conditionsArray:
            if i < 25:  # Working on cols
                result = self.compareLists(line, chuncksArray[1][
                    i])  # Returns a Tuple consisting of: [0] the number of chunks different
                # [1] A flag indicates if there are more than 8 chucks in a given row/col in the solution.
                if (result[1]):  # result[1] consists of the flag, (True or false). If Flag == True then...
                    compareConditionsArray[1][i] = 0.5 - 0.0625 * result[
                        0]  # If the flag is raised the reduction is harsher.
                else:
                    compareConditionsArray[1][i] = 0.5 - (0.0625 / 2) * result[0]
            else:  # Working on rows
                result = self.compareLists(line, chuncksArray[0][i - 25])
                if (result[1]):
                    compareConditionsArray[0][i - 25] = 0.5 - 0.0625 * result[0]
                else:
                    compareConditionsArray[0][i - 25] = 0.5 - (0.0625 / 2) * result[0]

                # compareConditionsArray[1][i-25] = len(set(chuncksArray[0][i]) & set(line))
            i += 1

        return compareConditionsArray

    def gradeForConditions(self, solution):  # Generating a grade according to the accuracy of the proposed solution
        # Creates a list containing 2 lists, each of 25 items, all set to 0
        # will be array for all grades for each column and row according to the conditions file order(meaning this array length will be 50)
        w, h = 25, 2
        gradesArray = [[0 for x in range(w)] for y in range(h)]
        # grade will be build by 50% comparePopulation and 50% chuncks comparasion
        comparePopulationArray = self.comparePopulation(solution)  # consists population differences
        compareConditionsArray = self.compareConditions(solution)  # consists grades already
        for i in range(0, 1):  # 0 for rows, 1 for cols
            for j in range(0, 25):
                absoulte_difference = abs(comparePopulationArray[i][j])
                if absoulte_difference == 0:
                    gradesArray[i][j] = 0.5 + compareConditionsArray[i][j]
                else:
                    gradesArray[i][j] = 0.5 - 0.5 * (absoulte_difference / 25) + compareConditionsArray[i][j]

        return gradesArray

    def gradeCalculator(self, gradesArray):  # Taking the data from the other functions, this one calculated the final grade.
        grade = 0
        for i in range(0, 1):
            for j in range(0, 25):
                grade += gradesArray[i][j]

        grade /= 25  # Calculating the average after going through the 2D array
        return grade  # A float

    def matrixGrade(self, solution):  # Starting a grading flow for a solution.
        grade = self.gradeCalculator(self.gradeForConditions(solution))  # Expecting to get a float to grade1
        return grade * 100

    def specificCrossOver(self, col, solution1, solution2):  # Creating a cross over between two cols.
        col1 = solution1[:, col]
        col2 = solution2[:, col]
        solution1[:, col] = np.copy(col2)
        solution2[:, col] = np.copy(col1)
        return solution1, solution2

    def optimizer(self, solution1, solution2):  # Our optimizer function, used in Darwin mode and LeMark mode.
        solution1Grade = self.matrixGrade(solution1)
        solution2Grade = self.matrixGrade(solution2)
        newSolution1Grade = solution1Grade
        newSolution2Grade = solution2Grade
        timeout = 0
        while (newSolution1Grade <= solution1Grade and newSolution2Grade <= solution2Grade):
            if (timeout >= 64):
                # print("NO SUCCESS........................................................")
                copySolution1 = solution1
                copySolution2 = solution2
                break
            col = random.randint(8, 32)
            copySolution1, copySolution2 = self.specificCrossOver(col, np.copy(solution1), np.copy(solution2))
            newSolution1Grade = self.matrixGrade(copySolution1)
            newSolution2Grade = self.matrixGrade(copySolution2)
            timeout += 1

        # if (timeout < 64):
        # print("SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return copySolution1, copySolution2