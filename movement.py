import random
import numpy as np
from numba import njit, jit

PREDATOR = "PREDATOR"
PREY = "PREY"

@njit
def moveRandomPredator(agent, grid, speed=1):
    currentCell = agent.currentPosition

    ## When we have multiple preys, predators are allowed to move to preys' cells ???
    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(currentCell)



    # move to selected unoccupied cell
    if len(unoccupiedNeighbourCells) != 0:
        randIndx = random.randint(0, len(unoccupiedNeighbourCells) - 1)
        grid.movePredatorToNewCell(agent, unoccupiedNeighbourCells[randIndx])

@njit
def predatorNextMove(predator, prey, grid):
    # get unoccupied neighbour cells of the prey
    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(prey.currentPosition)

    # claculate the distance from predator's position to each unoccupied neighbour cells
    dists = np.zeros(shape=(len(unoccupiedNeighbourCells)))
    if len(dists) == 0:
        moveRandomPredator(predator, grid)
    else:
        for cellIdx in range(len(unoccupiedNeighbourCells)):
            dists[cellIdx] = grid.getManhattenDistance(predator.currentPosition, unoccupiedNeighbourCells[cellIdx])

        # get the neighbour cell with the minimum distance as the targte destination of the predator
        destination = unoccupiedNeighbourCells[dists.argmin()]

        dx = (predator.currentPosition[1] - destination[1]) % grid.numberOfColumns
        dy = (predator.currentPosition[0] - destination[0]) % grid.numberOfRows
        if dx > (grid.numberOfColumns/2):
            dx = grid.numberOfColumns - dx

        if dy > (grid.numberOfRows/2):
            dy = grid.numberOfRows - dy

        if dx > dy:
            direction = grid.getXDirectionToMove(predator.currentPosition, destination)
        else:
            direction = grid.getYDirectionToMove(predator.currentPosition, destination)

        nextPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
        if grid.isCellOccupiedByPreyOrPredator(nextPosition) == False:
            grid.movePredatorToNewCell(predator, nextPosition)

        # else move randomly
        else:
           moveRandomPredator(predator, grid)