"""
Computational Biology: Third Assignment.
Developed by CUCUMBER an OrSN Company and Kfir Inbal
June 2021.
UNAUTHORIZED REPLICATION OF THIS WORK IS STRICTLY PROHIBITED.
"""
from sys import argv

# Basic grid design, courtesy of Auctux: https://www.youtube.com/watch?v=d73z8U0iUYE
# import matplotlib.pyplot
import pygame
import pygame.font
import numpy as np
import random

class Grid:
    def __init__(self, width, height, offset, screen):
        self.chuncks_flag = 0
        self.columns = 10
        self.rows = 10
        self.screen = screen
        self.scale = width / (self.columns)
        self.size = (self.rows, self.columns)
        self.offset = offset

        self.examples = []  # Stores tuples of matrix and its corresponding digit
        self.learningGroup = []
        self.index = 0
        self.weightsMatrix = []
        self.currentExample = 0
        self.currentGraph = []
        self.num_success_fixes = 0
        self.learningSetSize = int(argv[2])  # Accepted value [1,10]. Represent how many examples we learn per digit.
        self.gradualLearning = int(argv[3])  # Boolean value 0 or 1. If 1, We will learn an additional digit at a time.
        if self.gradualLearning == 1:
            self.learningStage = 0
        else:
            self.learningStage = 10

        self.examplesSize = 10


    #This function do all the learning process, reading from the given txt input of digits and creates self.learningGroup and self.examples
    def inputToData(self):
        self.learningGroup = []
        self.examples = []
        self.weightsMatrix = []
        file = open(argv[1], "r+")
        counter = 0  # Counts how many examples of the same digit we seen.
        currentDigit = 0  # Provides a proper label.
        matrix = []
        for line in file:
            row = []
            if len(line) < 10:  # If we encountered an empty line, we know that we have a complete matrix to add.
                matrix = np.array(matrix)
                if counter < self.learningSetSize:
                    self.learningGroup.append([matrix, currentDigit])
                if counter == 0:
                    for i in range(10):
                        self.examples.append([self.randomMatrixGenerator(np.copy(matrix), 10), currentDigit])
                counter += 1
                matrix = []

                if counter >= 10:
                    currentDigit += 1
                    counter = 0
                    if currentDigit > self.learningStage:
                        file.close()
                        self.weightsMatrix = self.weightsCalculator(self.learningGroup)
                        self.completeWeightsMatrix()
                        return
                continue
            for bit in line:
                try:
                    if int(bit) != 1:  # We need to check if we opted for -1 or 0.
                        row.append(int(argv[5]))
                    else:  # If the bit is 1, that's irrelevant.
                        row.append(int(bit))
                except:
                  continue
            matrix.append(row)

        if len(matrix) > 0:
            if self.learningSetSize > 9:
                matrix = np.array(matrix)
                self.learningGroup.append([matrix, currentDigit])

        file.close()

        self.weightsMatrix = self.weightsCalculator(self.learningGroup)
        self.completeWeightsMatrix()

    #This function displays the matrix in the self.currentGraph
    def drawGraph(self, off_color, on_color, surface):
        for x in range(self.rows):
            for y in range(self.columns):
                y_pos = y * self.scale
                x_pos = x * self.scale

                if self.currentGraph[y][x] == 1:
                    pygame.draw.rect(surface, on_color,
                                     [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
                else:  # 0 or -1.
                    pygame.draw.rect(surface, off_color,
                                     [x_pos, y_pos, self.scale - self.offset, self.scale - self.offset])
        pygame.display.update()

    #This function creates the flow of displays and fixing process
    def gridUpdater(self, off_color, on_color, surface):
        if self.currentExample >= len(self.examples):
            self.currentExample = len(self.examples) - 1

        self.currentGraph = self.examples[self.currentExample][0]
        self.drawGraph(off_color, on_color, surface) #showing example before fix
        self.currentGraph = self.fixNext()
        self.drawGraph(off_color, on_color, surface) #showing example after fix
        self.currentGraph = np.ones(shape=(10, 10))
        self.drawGraph(off_color, on_color, surface) #showing a black grid for separation between examples

    #This function do all the magic, trying to fix the matrixes in self.examples
    def fixNext(self):
        newMatrix = self.determineCell(np.copy(self.examples[self.currentExample][0]))
        timeout = 0
        while 1:
            distance = self.hammingDistance(np.copy(self.learningGroup[self.examples[self.currentExample][1]][0]),np.copy(newMatrix))
            if distance == 0: #if matrix is fully fixed
                self.num_success_fixes += 1
                break

            if timeout == 1000: #if distance still not 0 and it has been 1000 iterations then it breaks out of the loop
                if int(argv[4]) == 1: #Runs the optimization function if the given argument for optimizaion is 1
                    newMatrix = self.closestSolution(np.copy(self.examples[self.currentExample][0]))
                break
            newMatrix = self.determineCell(np.copy(newMatrix))
            timeout += 1


        self.currentExample += 1

        if self.currentExample >= len(self.examples): #if we tried to fix all given examples, it's time to check the success percentages :)
            success_percent = self.num_success_fixes / len(self.examples)
            if success_percent * 100 >= 90:
                if success_percent > 1:
                    success_percent = 1
                print(str(float(success_percent * 100)) + "% = SUCCESS!!!")
            else:
                print(str(float(success_percent * 100)) + "% = FAILURE!!!")

            if self.gradualLearning == 1: #if we are in gradual learning we want to fully stop when we learned all digits
                if self.learningStage == 10:
                    x = input("STOPPED")
                else: #continuing the loop, and learning another digit
                    self.learningStage += 1
                    self.inputToData()
            else:
                x = input("STOPPED")
        return newMatrix

    #This function flattens all matrixes in a given list of matrixes
    def flattenAllPatterns(self, allPatterns):
        for p in allPatterns:
            p[0] = p[0].flatten()
        return allPatterns

    #This function deflattens all matrixes in a given list of matrixes
    def deflatPatterns(self, allPatterns):
        for p in allPatterns:
            p[0] = np.array((p[0].reshape((10, 10))))
        return allPatterns

    #all patterns should have the same shape.
    def weightsCalculator(self, allPatterns):
        pShape = (allPatterns[0][0]).shape
        allPatterns = self.flattenAllPatterns(allPatterns)
        pShape = (1, pShape[0]*pShape[1])
        weightsMatrix = np.zeros((pShape[1], pShape[1]))
        for i in range(pShape[1] - 1):
            for j in range(i, pShape[1]):
                if i == j:
                    continue
                else:
                    sum = 0
                    for p in allPatterns:
                        if p[0][i] == p[0][j]:
                            sum += 1
                        else:
                            sum -= 1

                    weightsMatrix[i, j] = sum
        allPatterns = self.deflatPatterns(allPatterns)
        return weightsMatrix

    #This function completes the empty side of the weightsmatrix(the left side to the main diagonal) to be the mirror of the other side
    def completeWeightsMatrix(self, weightsMatrix = None):
        if weightsMatrix is None:
            weightsMatrix = self.weightsMatrix

        for x in range(len(weightsMatrix)):
            for y in range(len(weightsMatrix[x])):
                if x == y:
                    continue
                if weightsMatrix[x][y] != int(argv[5]):
                    weightsMatrix[y][x] = weightsMatrix[x][y]
                else:
                    weightsMatrix[x][y] = weightsMatrix[y][x]

    # This app gets a matrix and a cell (x,y) coordinate it needs to change and uses the weightsMatrix.
    # Using that information, the function extracts the line
    def determineCell(self, matrix, weightsMatrix = None):
        if weightsMatrix is None:
            weightsMatrix = self.weightsMatrix
        determiner = 0
        row = random.randint(0, 99)
        self.detectRow = row
        matrix = matrix.flatten()
        for y in range(100):
            determiner += (weightsMatrix[row][y] * matrix[y])
        if determiner >= 0:
            matrix[row] = 1
        else:
            matrix[row] = int(argv[5])
        matrix = np.array((matrix.reshape((10, 10))))
        return matrix

    # Calculates the Hamming Distance between an example and a solution.
    def hammingDistance(self, solution, matrixToCheck):
        distance = 0
        if(matrixToCheck.shape == (100,)):
            matrixToCheck = np.array((matrixToCheck.reshape((10, 10))))
        self.cellsFlip(matrixToCheck)
        solution = solution.flatten()
        matrixToCheck = matrixToCheck.flatten()  # (row,col) = (1,100)
        for index in range(len(solution)):
            if solution[index] != matrixToCheck[index]:  # If there's difference between two cells,
                # we increase the distance
                distance += 1
        return distance

    # In this function, we are looking for the solution that is the closest to our example.
    # This is part of the optimization flow.
    def closestSolution(self, matrix):
        winningLabel = 0
        hammingDistanceToBeat = 100
        for index in range(len(self.learningGroup)):
            hammingDistance = self.hammingDistance(self.learningGroup[index][0], matrix)
            if hammingDistance < hammingDistanceToBeat:
                winningLabel = index
                hammingDistanceToBeat = hammingDistance
        fixed_matrix = self.targetedSolution(matrix, self.learningGroup[winningLabel])

        return fixed_matrix

    #This function, as the final part of the optimization flow, creates a weights matrix only for the given solution
    #and then fix the given matrix according to that weights matrix
    def targetedSolution(self, matrix, closestSolution):
        matrix = matrix.flatten()
        allPatterns = [closestSolution]
        weights_matrix = self.weightsCalculator(allPatterns)
        self.completeWeightsMatrix(weights_matrix)
        timeout = 0
        fixed_matrix = matrix
        while True:
            distance = self.hammingDistance(np.copy(closestSolution[0]), np.copy(fixed_matrix))
            if distance == 0:
                self.num_success_fixes += 1
                break
            if timeout == 1000:
                break
            fixed_matrix = self.determineCell(fixed_matrix, weights_matrix)
            timeout += 1
        return fixed_matrix

    #This function generates matrix that is different in 10% of the bits from the given matrix
    def randomMatrixGenerator(self, matrix, numberOfChanges):
        for _ in range(numberOfChanges):
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            if matrix[x][y] == int(argv[5]):
                matrix[x][y] = 1
            else:
                matrix[x][y] = int(argv[5])
        return matrix

    #This function fixes cases where the matrix is inverse, meaning has a black background and white digit
    def cellsFlip(self, matrix):
        if matrix[0][0] == matrix[0][9] == matrix[9][0] == matrix[9][9] == 1:
            for x in range(10):
                for y in range(10):
                    if matrix[x][y] == 1:
                        matrix[x][y] = argv[5]  # 0 or -1
                    else:
                        matrix[x][y] = 1
        return matrix