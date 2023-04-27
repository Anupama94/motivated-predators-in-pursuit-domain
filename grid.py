import numpy as np
from numba import types, typeof, typed, njit
from numba.typed import Dict
from operator import itemgetter
from numba.extending import typeof_impl, type_callable, models, register_model
import numpy as np
from numba import int32, float32, deferred_type
from numba.experimental import jitclass
import math
from cell import Node


from cell import Cell

PREY = "PREY"
PREDATOR = "PREDATOR"

@njit
def getCellKeyName(row, col):
    return str(row) + '-' + str(col)

@njit
def getIndOfMinTuple(arrayOfTuples, keyInd):
    globalMin = math.inf
    globalMinInd = 0
    for i in range(len(arrayOfTuples)):
        if arrayOfTuples[i].fScore < globalMin:
            globalMinInd = i
            globalMin = arrayOfTuples[i].fScore
    return globalMinInd

@njit
def isTupleInArray(givenTuple, givenArray):
    for i in range(len(givenArray)):
        if givenArray[i][0] == givenTuple[0] and givenArray[i][1] == givenTuple[1] and givenArray[i][2] == givenTuple[2] and \
                i[3] == givenTuple[3] and givenArray[i][4] == givenTuple[4]:
            return True
    return False


spec = [

    ('numberOfRows', int32),               # a simple scalar field
    ('numberOfColumns', int32),
    # ('grid', types.ListType(Cell.class_type.instance_type)),          # an array field
    # ('grid', types.DictType(types.unicode_type, int32)),
    ('grid', int32[:, :, :]),
]



@jitclass(spec)
class Grid:

    def __init__(self, rowNum, colNum):
        self.numberOfRows = rowNum
        self.numberOfColumns = colNum
        self.grid = np.empty(shape=(rowNum, colNum, 4), dtype=int32)

        for row in range(rowNum):
            for col in range(colNum):
                # row, col, preyId, predatorId
                self.grid[row, col] = np.array([row, col, 0, 0], dtype=int32)

    def getCell(self, row, col):
        return self.grid[row, col]

    def occupyCellByPrey(self, row, col, prey):
        cell = self.getCell(row, col)
        cell[2] = prey.id
        prey.currentPosition = np.array([row, col], dtype=int32)

    def occupyCellByPredator(self, row, col, predator):
        cell = self.getCell(row, col)
        cell[3] = predator.id
        predator.currentPosition = np.array([row, col], dtype=int32)

    # # change in direction x mens change in column number
    def getDirectDeltaX(self, currentCell, targetCell):
        return (currentCell[1] - targetCell[1]) % self.numberOfColumns

    def getDirectDeltaY(self, currentCell, targetCell):
        return (currentCell[0] - targetCell[0]) % self.numberOfRows

    def getIndirectDeltaX(self, currentCell, targetCell):
        return (targetCell[1] - currentCell[1]) % self.numberOfColumns

    def getIndirectDeltaY(self, currentCell, targetCell):
        return (targetCell[0] - currentCell[0]) % self.numberOfRows

    def getXDirectionToMove(self, currentCell, targetCell):
        directDeltaCol = self.getDirectDeltaX(currentCell, targetCell)
        indirectDeltaCol = self.getIndirectDeltaX(currentCell, targetCell)

        if directDeltaCol < indirectDeltaCol:
            return "left"
        else:
            return "right"

    def getYDirectionToMove(self, currentCell, targetCell):
        directDeltaY = self.getDirectDeltaY(currentCell, targetCell)
        indirectDeltaY = self.getIndirectDeltaY(currentCell, targetCell)

        if directDeltaY < indirectDeltaY:
            return "up"
        else:
            return "down"

    def getNeighbours(self, currentLocation):
        cellRow = currentLocation[0]
        cellCol = currentLocation[1]
        neighbours = np.zeros(shape=(4, 4), dtype=int32)

        neighbours[0] = self.getCell(cellRow, (cellCol - 1) % self.numberOfColumns)


        neighbours[1] = self.getCell((cellRow - 1) % self.numberOfRows, cellCol)

        neighbours[2] = self.getCell(cellRow, (cellCol + 1) % self.numberOfColumns)

        neighbours[3] = self.getCell((cellRow + 1) % self.numberOfRows, cellCol)

        return neighbours

    def isCellOccupiedByPreyOrPredator(self, currentLocation):
        currentCell = self.getCell(currentLocation[0], currentLocation[1])
        if currentCell[2] != 0 or currentCell[3] != 0:
            return True
        return False

    def isCellOccupiedByPredator(self, currentCell):
        return currentCell[3] != 0

    def isCellOccupiedByPrey(self, currentCell):
        return currentCell[2] != 0

    def getUnoccupiedNeighbourCellCount(self, currentLocation, occupiedBy=None):

        count = 0

        neighbours = self.getNeighbours(currentLocation)

        for cellInd in range(len(neighbours)):
            cell = neighbours[cellInd]

            if occupiedBy == PREDATOR:
                if self.isCellOccupiedByPredator(cell) == False:
                    count += 1
            elif occupiedBy == PREY:
                if self.isCellOccupiedByPrey(cell) == False:
                    count += 1
            else:
                if self.isCellOccupiedByPreyOrPredator(cell) == False:
                    count += 1

        return count

    def getUnoccupiedNeighbourCells(self, currentLocation, occupiedBy=None):
        count = self.getUnoccupiedNeighbourCellCount(currentLocation, occupiedBy)

        unoccupiedNeighbourCells = np.zeros(shape=(count, 4), dtype=int32)

        neighbours = self.getNeighbours(currentLocation)

        counter = 0
        for cellInd in range(len(neighbours)):
            cell = neighbours[cellInd]

            if occupiedBy == PREDATOR:
                if self.isCellOccupiedByPredator(cell) == False:
                    unoccupiedNeighbourCells[counter] = cell
                    counter += 1

            elif occupiedBy == PREY:
                if self.isCellOccupiedByPrey(cell) == False:
                    unoccupiedNeighbourCells[counter] = cell
                    counter += 1
            else:
                if self.isCellOccupiedByPreyOrPredator(cell) == False:
                    unoccupiedNeighbourCells[counter] = cell
                    counter += 1

        return unoccupiedNeighbourCells

    def movePreyToNewCell(self, prey, newCell):
        oldCell = self.getCell(prey.currentPosition[0], prey.currentPosition[1])
        oldCell[2] = 0
        self.occupyCellByPrey(newCell[0], newCell[1], prey)

    def movePredatorToNewCell(self, predator, newCell):
        oldCell = self.getCell(predator.currentPosition[0], predator.currentPosition[1])
        oldCell[3] = 0
        self.occupyCellByPredator(newCell[0], newCell[1], predator)

    def getLeftCell(self, currentCell, step=1):
        leftCell = self.grid[currentCell[0], (currentCell[1] - step) % self.numberOfColumns]
        return leftCell

    def getRightCell(self, currentCell, step=1):
        rightCell = self.grid[currentCell[0], (currentCell[1] + step) % self.numberOfColumns]
        return rightCell

    def getTopCell(self, currentCell, step=1):
        topCell = self.grid[(currentCell[0] - step) % self.numberOfRows, currentCell[1]]
        return topCell

    def getDownCell(self, currentCell, step=1):
        downCell = self.grid[(currentCell[0] + step) % self.numberOfRows, currentCell[1]]
        return downCell

    def getCellInGivenDirection(self, currentCell, direction, step=1):
        if direction == "left":
            return self.getLeftCell(currentCell, step)

        if direction == "up":
            return self.getTopCell(currentCell, step)

        if direction == "right":
            return self.getRightCell(currentCell, step)

        if direction == "down":
            return self.getDownCell(currentCell, step)

        if direction == "":
            return currentCell
    #
    # # https://blog.demofox.org/2017/10/01/calculating-the-distance-between-points-in-wrap-around-toroidal-space/
    def getManhattenDistance(self, currentCell, targetLocation):
        dy = np.abs(currentCell[0] - targetLocation[0])
        dx = np.abs(currentCell[1] - targetLocation[1])

        if dx > (self.numberOfColumns / 2):
            dx = self.numberOfColumns - dx

        if dy > (self.numberOfRows / 2):
            dy = self.numberOfRows - dy

        return dx + dy


    # https://www.raywenderlich.com/3016-introduction-to-a-pathfinding
    def aStarPathFinding(self, startingCell, targetCell):
            #(currentCell, parentCell, g, h, f)
            opened = []
            closed = []
            gScore = 0
            hScore = self.getManhattenDistance(startingCell, targetCell)
            fScore = gScore + hScore

            startingNode = (startingCell[0], startingCell[1], startingCell[0], startingCell[1], gScore, hScore, fScore)
            opened.append(startingNode)
            closed.append(startingNode)
            i = 0
            exited = False
            while i < 100 and len(opened) > 0:
                i += 1
                # select the node in open with the lowest fScore as the current node
                globalMin = math.inf
                globalMinInd = 0
                for idx in range(len(opened)):
                    if opened[idx][6] <= globalMin:
                        globalMinInd = idx
                        globalMin = opened[idx][6]
                currentNode = opened[globalMinInd]
                # add current node to closed
                if i == 1:
                    closed.remove(startingNode)
                closed.append(currentNode)
                # remove current node from open
                opened.remove(currentNode)

                if currentNode[0] == targetCell[0] and currentNode[1] == targetCell[1]:
                    exited = True
                    break

                neighboursOfCurrent = self.getNeighbours([currentNode[0], currentNode[1]])
            #
                for neighbourIdx in range(4):
                    neighbour = neighboursOfCurrent[neighbourIdx]
                    gScore = currentNode[4] + 1
                    hScore = 1
                    fScore = gScore + hScore

                    neighbourNode = (neighbour[0], neighbour[1], currentNode[0], currentNode[1], gScore, hScore, fScore)
            # #
                    existsInClosed = False

                    for abcIdx in range(len(closed)):
                        # print("closed IDX", closed[abcIdx], neighbourNode)
                        if closed[abcIdx][0] == neighbourNode[0] and closed[abcIdx][1] == neighbourNode[1] and closed[abcIdx][2] == neighbourNode[2] and\
                                closed[abcIdx][3] == neighbourNode[3] and closed[abcIdx][4] == neighbourNode[4] and closed[abcIdx][5] == neighbourNode[5] and\
                                closed[abcIdx][6] == neighbourNode[6]:
                            existsInClosed = True
                    if self.isCellOccupiedByPreyOrPredator(neighbour) or existsInClosed == True:

                        continue

            #
            #
                    # is new path to neighbour is shorter or neightbour is not in open
            #
                    existsInOpen = False
                    for ijk in range(len(opened)):
                        if opened[ijk][0] == neighbourNode[0] and opened[ijk][1] == neighbourNode[1] and opened[ijk][2] == neighbourNode[2] and \
                                opened[ijk][3] == neighbourNode[3] and opened[ijk][4] == neighbourNode[4] and \
                                opened[ijk][5] == neighbourNode[5] and \
                                opened[ijk][6] == neighbourNode[6]:
                            existsInOpen = True

                    if existsInOpen == False or fScore < currentNode[4]:
                        if existsInOpen == False:
                            opened.append(neighbourNode)

            # trace back the shortest path
            node = closed[len(closed)-1]
            bogusVal = (100, 100)
            finalPath = [bogusVal]


            if not exited:
                return finalPath
            else:
                finalPath.remove(bogusVal)
                while True:


                    if node[0] == startingCell[0] and node[1] == startingCell[1]:
                        break

                    finalPath.append((node[0], node[1]))
                    for parentIdx in range(len(closed)):
                        if node[2] == closed[parentIdx][0] and node[3] == closed[parentIdx][1]:
                            node = closed[parentIdx]
                            break

                return finalPath


    #
    # def seek(self, predator, distance):
    #     detectedPreyCells = []
    #     currentRow = predator.currentPosition.row
    #     currentCol = predator.currentPosition.col
    #     for j in range(distance+1):
    #         for i in range(j+1):
    #             leftLowerCell = self.getCell((currentRow - (j - i)) % self.numberOfRows, (currentCol - i) % self.numberOfColumns)
    #             rightLowerCell = self.getCell((currentRow - (j - i)) % self.numberOfRows, (currentCol + i) % self.numberOfColumns)
    #             leftUpperCell = self.getCell((currentRow + (j - i)) % self.numberOfRows, (currentCol - i) % self.numberOfColumns)
    #             rightUpperCell = self.getCell((currentRow + (j - i)) % self.numberOfRows, (currentCol + i) % self.numberOfColumns)
    #
    #             # print("left lower row", leftLowerCell.getRow())
    #             # print("left lower col", leftLowerCell.getCol())
    #             #
    #             # print("right lower row", rightLowerCell.getRow())
    #             # print("right lower col", rightLowerCell.getCol())
    #             #
    #             # print("left upper row", leftUpperCell.getRow())
    #             # print("left upper col", leftUpperCell.getCol())
    #             #
    #             # print("right upper row", rightUpperCell.getRow())
    #             # print("right upper col", rightUpperCell.getCol())
    #             if leftLowerCell.value[PREY] != 0:
    #                 if leftLowerCell not in detectedPreyCells:
    #                     detectedPreyCells.append(leftLowerCell)
    #             elif rightLowerCell.value[PREY] != 0:
    #                 if rightLowerCell not in detectedPreyCells:
    #                     detectedPreyCells.append(rightLowerCell)
    #             elif leftUpperCell.value[PREY] != 0:
    #                 if leftUpperCell not in detectedPreyCells:
    #                     detectedPreyCells.append(leftUpperCell)
    #             elif rightUpperCell.value[PREY] != 0:
    #                 if rightUpperCell not in detectedPreyCells:
    #                     detectedPreyCells.append(rightUpperCell)
    #     return detectedPreyCells
    #
    def markDeadPrey(self, prey):
        currentCell = self.getCell(prey.currentPosition[0], prey.currentPosition[1])
        currentCell[2] = 0
        prey.status = "DEAD"
    #
    # def seekPreysInVisionRange(self, predator, preys, visionRange):
    #     detectedPreys = []
    #     for prey in preys:
    #         distance = self.getManhattenDistance(predator.currentPosition, prey.currentPosition)
    #         if distance <= visionRange:
    #             detectedPreys.append(prey)
    #
    #     # returns an array of arrays
    #     return detectedPreys


