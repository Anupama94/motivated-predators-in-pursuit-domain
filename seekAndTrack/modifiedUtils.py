import math
import random
from numba import njit
import numpy as np
from agent import Prey, Predator
from modifiedIncentiveFunctions import calculateLocalIncentive, calculateLocalEfficiencyIncentive


@njit
def map_difficulty_level(x):
    if x == 0:
        return "easy"
    elif x == 1:
        return "moderate"
    else:
        return "difficult"

@njit
def f_plain(x):
    return x * (x - 1)


@njit
def integrate_f_numba(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_plain(a + i * dx)
    return s * dx


@njit
def apply_integrate_f_numba(col_a, col_b, col_N):
    n = len(col_N)
    result = np.empty(n, dtype='float64')
    assert len(col_a) == len(col_b) == n
    for i in range(n):
        result[i] = integrate_f_numba(col_a[i], col_b[i], col_N[i])
    return result

@njit
def formatFolderName(ratio):
    name = ""
    if len(ratio) == 3:
        name = str(ratio[0]) + "-" + str(ratio[1]) + "-" + str(ratio[2])
    return name

@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """

    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit
def configurePreys(numberOfPreys, difficulty):
    # preys get negative ids stating from -1
    preys = []
    for p in range(numberOfPreys):
        id = (p + 1) * -1

        newPrey = Prey(id)
        # easy
        if difficulty == 0:
            newPrey.sig = rand_choice_nb(np.array([2, 4, 6]), np.array([0.8, 0.1, 0.1]))
        # moderately difficult
        elif difficulty == 1:
            newPrey.sig = rand_choice_nb(np.array([2, 4, 6]), np.array([1/3, 1/3, 1/3]))
        # difficult
        else:
            newPrey.sig = rand_choice_nb(np.array([2, 4, 6]), np.array([0.1, 0.1, 0.8]))

        if newPrey.sig == 2:
            newPrey.stamina = 0.3
        elif newPrey.sig == 4:
            newPrey.stamina = 0.2
        else:
            newPrey.stamina = 0.1
        preys.append(newPrey)

    return preys

@njit
def configurePredators(numberOfPredators, ratio):
    predators = []

    for p in range(numberOfPredators):
        idInInt = p + 2
        id = idInInt
        predators.append(Predator(id))
        aff = ratio[0]
        pow = ratio[1]
        if idInInt <= (aff + 1):
            predators[p].policy = "affiliation"
        elif idInInt <= aff + pow + 1:
            predators[p].policy = "power"
        else:
            predators[p].policy = "achievement"

    return predators

# only nxn grids are allowed
@njit
def assignPosition(occupied, gridSize, counter):
    sizeUpperBound = gridSize - 1
    row = random.randint(0, sizeUpperBound)
    col = random.randint(0, sizeUpperBound)
    position = [row, col]

    while True:
        for i in range(counter):
            if occupied[i, 0] == position[0] and occupied[i, 1] == position[1]:
                row = random.randint(0, sizeUpperBound)
                col = random.randint(0, sizeUpperBound)
                position = [row, col]
                break
        occupied[counter] = position
        return (row, col, occupied)



@njit
def getClosestTargetCell(grid, possibleTargetCells, currentPredator):
    shortestDist = math.inf
    closestCellInd = 0
    for cellInd in range(len(possibleTargetCells)):
        cell = possibleTargetCells[cellInd]
        dist = grid.getManhattenDistance(currentPredator.currentPosition, cell)
        if dist < shortestDist:
            shortestDist = dist
            closestCellInd = cellInd

    return possibleTargetCells[closestCellInd]

# this is patch for aStarPathFinding algo
def targetIsApproachable(currentPredator, grid, possibleTargetCells):
    unoccupiedNeighboursOfCurrentCell = grid.getUnoccupiedNeighbourCells(
        currentPredator.currentPosition)

    # no possibleTargetCells means the prey has no unoccupied cells around itself
    # no unoccupiedNeighboursOfCurrentCell means that the predator cant move coz its been surrounded by obstacles
    if len(unoccupiedNeighboursOfCurrentCell) == 0 or len(possibleTargetCells) == 0:
        return False
    return True


@njit
def modified_calculateLocalIncentivesForEachGoal(prey,
                                                 predators, grid, significance_weight, adjacent_weight,predatorAffiliationRange):
    predatorCount = len(predators)
    normalizedS = prey.sig / 6

    a = 0
    for pred in predators:
        dist = grid.getManhattenDistance(pred.currentPosition, prey.currentPosition)
        if dist <= predatorAffiliationRange:
            a += 1

    normalizedA = a / predatorCount # divide by maximum number of predators that the predatorAffiliationRange
    return calculateLocalIncentive(normalizedS, normalizedA, significance_weight, adjacent_weight, a)


# regardless of the incentive, targetting which prey is the most feasible?
# feasibility is defined by the distance to the prey
@njit
def modified_localEfficiencyIncentive(prey, grid, currentPredator, thresholdDistance):
    d = grid.getManhattenDistance(currentPredator.currentPosition, prey.currentPosition)

    return calculateLocalEfficiencyIncentive(d, grid.numberOfRows, thresholdDistance)