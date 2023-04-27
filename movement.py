import random
import numpy as np
from numba import njit, jit

PREDATOR = "PREDATOR"
PREY = "PREY"


def moveRandomPrey(agent, grid, speed=1):
    currentCell = agent.currentPosition

    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(currentCell, "")

    randIndx = random.randint(0, len(unoccupiedNeighbourCells))

    # if index is within array range, pick a cell to mpve to randomly
    # else do not move
    if randIndx < len(unoccupiedNeighbourCells):
        grid.moveToNewCell(agent, unoccupiedNeighbourCells[randIndx])

def moveRandomFastPrey(agent, grid, speed):
    currentCell = agent.currentPosition

    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCellsMultipleSteps(currentCell, speed)

    randIndx = random.randint(0, len(unoccupiedNeighbourCells))

    # if index is within array range, pick a cell to mpve to randomly
    # else do not move
    if randIndx < len(unoccupiedNeighbourCells):
        grid.moveToNewCell(agent, unoccupiedNeighbourCells[randIndx])

@njit
def moveRandomPredator(agent, grid, speed=1):
    currentCell = agent.currentPosition

    ## When we have multiple preys, predators are allowed to move to preys' cells ???
    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(currentCell)



    # move to selected unoccupied cell
    if len(unoccupiedNeighbourCells) != 0:
        randIndx = random.randint(0, len(unoccupiedNeighbourCells) - 1)
        grid.movePredatorToNewCell(agent, unoccupiedNeighbourCells[randIndx])


def moveAwayFromAllPredators(prey, predators, grid):
    unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(prey.currentPosition)

    dists = [prey.currentPosition]
    total = [0]
    for cell in unoccupiedNeighbourCells:
        totalDistance = 0
        for predator in predators:
            totalDistance += grid.getManhattenDistance(cell, predator.currentPosition)
        total.append(totalDistance)
        dists.append(cell)

    destination = dists[total.index(max(total))]

    grid.moveToNewCell(prey, destination)

def moveAwayFromClosestPredator(prey, predators, grid):
    distanceToPredators = []
    for predator in predators:
        distanceToPredators.append(grid.getManhattenDistance(prey.currentPosition, predator.currentPosition))

    cellToTurnAwayFrom = predators[distanceToPredators.index(min(distanceToPredators))].currentPosition

    dx = (prey.currentPosition.getCol() - cellToTurnAwayFrom.getCol()) % grid.numberOfColumns
    dy = (prey.currentPosition.getRow() - cellToTurnAwayFrom.getRow()) % grid.numberOfRows
    if dx > (grid.numberOfColumns / 2):
        dx = grid.numberOfColumns - dx

    if dy > (grid.numberOfRows / 2):
        dy = grid.numberOfRows - dy

    if dx > dy:
        direction = grid.getXDirectionToMove(predator.currentPosition, cellToTurnAwayFrom)
    else:
        direction = grid.getYDirectionToMove(predator.currentPosition, cellToTurnAwayFrom)

    nextPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
    if grid.isCellOccupied(nextPosition) == False:
        grid.moveToNewCell(predator, nextPosition)

    # else move randomly
    else:
        moveRandomPredator(predator, grid)

# used when A star fails to find a move in a given number of iterations
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

# for predator
def moveGreedy(predator, prey, grid):
    left, top, right, bottom = grid.getNeighbours(prey.currentPosition)
    # if laready in the neighbouring cell go to the prey's cell
    if (predator.currentPosition == left or
        predator.currentPosition == top or
        predator.currentPosition == right or
        predator.currentPosition == bottom) \
            and grid.isCellOccupied(prey.currentPosition) == False:

        # print("don't move")
        grid.moveToNewCell(predator, prey.currentPosition)


    else:
        # get unoccupied neighbour cells of the prey
        unoccupiedNeighbourCells = grid.getUnoccupiedNeighbourCells(prey.currentPosition)

        # claculate the distance from predator's position to each unoccupied neighbour cells
        dists = []
        for cell in unoccupiedNeighbourCells:
            dists.append(grid.getManhattenDistance(predator.currentPosition, cell))

        # get the neighbour cell with the minimum distance as the targte destination of the predator
        destination = unoccupiedNeighbourCells[dists.index(min(dists))]

        dx = (predator.currentPosition.getCol() - destination.getCol()) % grid.numberOfColumns
        dy = (predator.currentPosition.getRow() - destination.getRow()) % grid.numberOfRows
        if dx > (grid.numberOfColumns/2) :
            dx = grid.numberOfColumns - dx

        if dy > (grid.numberOfRows/2) :
            dy = grid.numberOfRows - dy

        if dx > dy:
            direction = grid.getXDirectionToMove(predator.currentPosition, destination)
        else:
            direction = grid.getYDirectionToMove(predator.currentPosition, destination)

        nextPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
        if grid.isCellOccupied(nextPosition) == False:
            grid.moveToNewCell(predator, nextPosition)

        # else move randomly
        else:
           moveRandomPredator(predator, grid)



def moveGreedy2(predator, prey, grid):
    # If already neighbouring the prey, try to move onto the prey so that if it  moves, the predator will follow
    left, top, right, bottom = grid.getNeighbours(prey.currentPosition)
    if (predator.currentPosition == left or
        predator.currentPosition == top or
        predator.currentPosition == right or
        predator.currentPosition == bottom)\
            and not grid.isCellOccupied(prey.currentPosition):
       grid.moveToNewCell(predator, prey.currentPosition)

    else:
        unoccupiedNeighbours = grid.getUnoccupiedNeighbourCells(prey.currentPosition)

        # Choose the nearest unoccupied cell neighbouring the prey as the destination
        directDists = []
        for unoccupiedNeighbour in unoccupiedNeighbours:
            sqauredDeltaX = np.square(grid.getDirectDeltaX(predator.currentPosition, unoccupiedNeighbour))
            squaredDeltaY = np.square(grid.getDirectDeltaY(predator.currentPosition, unoccupiedNeighbour))

            # x **2 + y ** 2
            summedSquared = sqauredDeltaX + squaredDeltaY

            # sqrt(x **2 + y ** 2)
            dist = np.sqrt(summedSquared)
            directDists.append(dist)

        indirectDists = []
        for unoccupiedNeighbour in unoccupiedNeighbours:
            sqauredDeltaX = np.square(grid.getIndirectDeltaX(predator.currentPosition, unoccupiedNeighbour))
            squaredDeltaY = np.square(grid.getIndirectDeltaY(predator.currentPosition, unoccupiedNeighbour))

            # x **2 + y ** 2
            summedSquared = sqauredDeltaX + squaredDeltaY

            # sqrt(x **2 + y ** 2)
            dist = np.sqrt(summedSquared)
            indirectDists.append(dist)

        # get the minimum value
        minDirectDistance = min(directDists)
        minIndirectDistance = min(indirectDists)

        destination = None
        if minDirectDistance > minIndirectDistance:
            # get the index of the minimum value
            minIndex = indirectDists.index(minIndirectDistance)
            destination = unoccupiedNeighbours[minIndex]
        else:
            minIndex = directDists.index(minDirectDistance)
            destination = unoccupiedNeighbours[minIndex]

        # delta x and delta y of destination neighbour cell and current position of the predator
        minDeltaX = grid.getMinDeltaX(predator.currentPosition, destination)
        maxDeltaX = grid.getMaxDeltaX(predator.currentPosition, destination)

        minDeltaY = grid.getMinDeltaY(predator.currentPosition, destination)
        maxDeltaY = grid.getMaxDeltaY(predator.currentPosition, destination)

        moved = False
        if maxDeltaX > maxDeltaY:
            direction = grid.getXDirectionToMove(predator.currentPosition, destination)
            # check whether its occupied !!!!!!! if not do
            newPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
            if grid.isCellOccupied(newPosition) == False:
                moved = True
                grid.moveToNewCell(predator, newPosition)

        if moved == False and maxDeltaX <= maxDeltaY:
            direction = grid.getYDirectionToMove(predator.currentPosition, destination)
            newPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
            # check whether its occupied !!!!!!! if not do
            if grid.isCellOccupied(newPosition) == False:
                moved = True
                grid.moveToNewCell(predator, newPosition)

        if moved == False:
            if minDeltaX < minDeltaY:
                direction = grid.getXDirectionToMove(predator.currentPosition, destination)
                newPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
                if grid.isCellOccupied(newPosition) == False:
                    moved = True
                    grid.moveToNewCell(predator, newPosition)

            else:
                direction = grid.getYDirectionToMove(predator.currentPosition, destination)
                newPosition = grid.getCellInGivenDirection(predator.currentPosition, direction)
                # check whether its occupied !!!!!!! if not do
                if grid.isCellOccupied(newPosition) == False:
                    moved = True
                    grid.moveToNewCell(predator, newPosition)

        if moved == False:
                moveRandom(predator, grid)


# for predators. All predators are updated at once
def moveTeammateAware(predators, prey, grid):
    # Calculate the distance from each predator to each cell neighboring the prey
    # left, top, right, down
    neighbourCells = grid.getNeighbours(prey.currentPosition)

    predatorDistances = []
    for predator in predators:
        dists = []
        for cell in neighbourCells:
            dist = grid.getManhattenDistance(predator.currentPosition, cell)
            dists.append(dist)
        predatorDistances.append(dists)

    predatorMaxDist = []
    for i in range(len(predatorDistances)):
        maxDist = min(predatorDistances[i])
        predatorMaxDist.append((i, maxDist))

    # Order the predators based on worst shortest distance to a cell neighboring the prey
    predatorMaxDist.sort(key=lambda x: x[1], reverse=True)

    # In order, the predators are assigned the unchosen destination that is closest to them
    currentlySelectedNeighbourCells = []
    for i in range(len(predatorMaxDist)):
        selectedPredator = predators[predatorMaxDist[i][0]]
        chosenDestination = None

        foundUnoccupiedCell = False
        directions = [0, 1, 2, 3]

        while foundUnoccupiedCell == False:
            predatorNumber = predatorMaxDist[i][0]
            minDist = min(predatorDistances[predatorNumber])

            originalIndex = predatorDistances[predatorNumber].index(minDist)
            indx = directions[originalIndex]

            if neighbourCells[indx] not in currentlySelectedNeighbourCells:
                chosenDestination = neighbourCells[indx]
                currentlySelectedNeighbourCells.append(neighbourCells[indx])
                break
            else:
                # if len(predatorDistances[predatorNumber]) == 0:

                predatorDistances[predatorNumber].remove(minDist)
                directions.remove(directions[originalIndex])

        # If the predator is already at the destination, try to move onto the prey so that if it moves, the predator will follow

        if selectedPredator.currentPosition == chosenDestination:
            # print(grid.isCellOccupied(chosenDestination))
            # grid.moveToNewCell(selectedPredator, prey.currentPosition)
            pass

        else:

            # Otherwise, use A* path planning to select a path, treating other agents as static obstacles
            shortestPath = grid.aStarPathFinding(selectedPredator.currentPosition, chosenDestination)

            # last index in the shortest path array contains the immediate next cell the predator have to move to
            if len(shortestPath) != 0 and grid.isCellOccupiedByPredator(shortestPath[len(shortestPath)-1]) == False:
                grid.moveToNewCell(selectedPredator, shortestPath[len(shortestPath)-1])
            # else:
            #
            #     moveRandom(selectedPredator, grid)


# for predators. All predators are updated at once
def moveTeammateAwareWithoutPredatorsAndPreysOcuupyingSameCell(predators, prey, grid):
    # Calculate the distance from each predator to each cell neighboring the prey
    # left, top, right, down
    directions = ["left", "up", "right", "down"]
    neighbourCells = grid.getNeighbours(prey.currentPosition)

    predatorDistances = []
    for predator in predators:
        dists = []
        for cell in neighbourCells:
            dist = grid.getManhattenDistance(predator.currentPosition, cell)
            dists.append(dist)
        predatorDistances.append(dists)

    predatorMaxDist = []
    for i in range(len(predatorDistances)):
        maxDist = min(predatorDistances[i])
        predatorMaxDist.append((i, maxDist))

    # Order the predators based on worst shortest distance to a cell neighboring the prey
    predatorMaxDist.sort(key=lambda x: x[1], reverse=True)

    # In order, the predators are assigned the unchosen destination that is closest to them
    for i in range(len(predatorMaxDist)):
        selectedPredator = predators[predatorMaxDist[i][0]]
        chosenDestination = None
        foundUnoccupiedCell = False
        directions = [0, 1, 2, 3]

        while foundUnoccupiedCell == False:
            predatorNumber = predatorMaxDist[i][0]
            minDist = min(predatorDistances[predatorNumber])

            originalIndex = predatorDistances[predatorNumber].index(minDist)
            indx = directions[originalIndex]

            if grid.isCellOccupied(neighbourCells[indx]) == False or selectedPredator.currentPosition == neighbourCells[indx]:
                chosenDestination = neighbourCells[indx]
                foundUnoccupiedCell = True
                break
            else:
                # if len(predatorDistances[predatorNumber]) == 0:

                predatorDistances[predatorNumber].remove(minDist)
                directions.remove(directions[originalIndex])

        # If the predator is already at the destination, try to move onto the prey so that if it moves, the predator will follow

        # if selectedPredator.currentPosition == chosenDestination and grid.isCellOccupied(prey.currentPosition) == False:
        #     print(grid.isCellOccupied(chosenDestination))
        #     grid.moveToNewCell(selectedPredator, prey.currentPosition)
        #
        # else:

        # Otherwise, use A* path planning to select a path, treating other agents as static obstacles
        shortestPath = grid.aStarPathFinding(selectedPredator.currentPosition, chosenDestination)

        # last index in the shortest path array contains the immediate next cell the predator have to move to
        if len(shortestPath) != 0:
            grid.moveToNewCell(selectedPredator, shortestPath[len(shortestPath)-1])
            # else:
            #
            #     moveRandom(selectedPredator, grid)-


