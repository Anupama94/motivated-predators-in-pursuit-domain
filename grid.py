from numba import njit
import numpy as np
from numba import int32
from numba.experimental import jitclass
import math

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

    def markDeadPrey(self, prey):
        currentCell = self.getCell(prey.currentPosition[0], prey.currentPosition[1])
        currentCell[2] = 0
        prey.status = "DEAD"


