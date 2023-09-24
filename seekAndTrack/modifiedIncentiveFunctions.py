import random
import numpy as np
import math
# from scipy.integrate import si
from scipy import LowLevelCallable
import numba
from numba import vectorize, int32, njit, cfunc, carray, float64, intc, types, jit

@njit
def getSumOfArray(array):
    totalSum = 0
    for i in range(len(array)):
        totalSum += array[i]
    return totalSum

@njit
def trapezoidal(f, a, b, n=1000):
    h = float(b-a)/n
    result = 0.5*f(a) + 0.5*f(b)
    for i in range(1, n):
        result += f(a + i*h)
    result *= h
    return result
@njit
def getModifiedPowerMotivationTendency(numberOfGoals):
    ix = np.linspace(0, 1, numberOfGoals)
    motivation = np.zeros((numberOfGoals), dtype=float64)

    def f(x):
        # dominant-only motive profile
        return 4 / (1 + math.exp(-20 * (x - 0.7))) - 4 / (1 + math.exp(-20 * (x - 0.9)))

    for i in range(len(ix)):
        # res = trapezoidal(f, ix[i], ix[i+1])
        res = f(ix[i])
        motivation[i] = res

    motivationalTendency = []
    for i in range(len(motivation)):
        yy = motivation[i] / getSumOfArray(motivation)
        motivationalTendency.append(yy)
    return motivationalTendency

@njit
def getModifiedAffiliationMotivationTendency(numberOfGoals):
    ix = np.linspace(0, 1, numberOfGoals)
    motivation = np.zeros((numberOfGoals), dtype=float64)

    def f(x):
        # combo motive profile
        return 4 / (1 + math.exp(-20 * (0.3 - x))) - 4 / (1 + math.exp(-20 * (0.1 - x)))

    for i in range(len(ix)):
        # res, err = si.quad(integrand, 1, args=(1.0,))
        # res = trapezoidal(f, ix[i], ix[i+1])
        res = f(ix[i])
        motivation[i] = res

    motivationalTendency = []
    for i in range(len(motivation)):
        yy = motivation[i] / getSumOfArray(motivation)
        motivationalTendency.append(yy)

    return motivationalTendency

@njit
def getModifiedAchievementMotivationTendency(numberOfGoals):
    ix = np.linspace(0, 1, numberOfGoals)
    motivation = np.zeros((numberOfGoals), dtype=float64)

    def f(x):
        # combo motive profile
        return 4 / (1 + math.exp(-20 * ((1 - x) - 0.4))) - 4 / (1 + math.exp(-20 * ((1 - x) - 0.6)))

    for i in range(len(ix)):
        # res, err = si.quad(integrand, 1, args=(1.0,))
        # res = trapezoidal(f, ix[i], ix[i+1])
        res = f(ix[i])
        motivation[i] = res

    motivationalTendency = []
    for i in range(len(motivation)):
        yy = motivation[i] / getSumOfArray(motivation)
        motivationalTendency.append(yy)

    return motivationalTendency


@njit
def calculateLocalIncentive(normalizedS, normalizedA, significanceWeight, adjacentWeight, p):
    if p < 1:
        return 0.6 + 0.025 * math.exp(-(1 - normalizedS)) * math.exp(2.5 * (1 - normalizedA))
    elif p >= 1 and p <= 3:
        return 0.4 + 0.025 * math.exp(-(1-normalizedS)) * math.exp(2.5*(1 - normalizedA))
    else:
      return 0.05 * math.exp(-(1 - normalizedS)) * math.exp(2.5 * (1 - normalizedA))

@njit
def calculateLocalEfficiencyIncentive(d, gridSize, threshold_d=5):

    if d <= 5:
        return 1 / (1 + math.exp(-0.8 * (threshold_d - d)))
    else:
        return 0.001

@njit
def calculateModifiedGlobalIncentive(preyVicinityRange, prey, predators, grid, numberOfPreys, currentPredator):
    # preyVicinityRange = 3 # hard coded for now
    numberOfPredatorsInVicinity = 0
    for predIdx in range(len(predators)):
        pred = predators[predIdx]
        # for the team incentive we consider the number of agents around a prey
        dist = grid.getManhattenDistance(pred.currentPosition, prey.currentPosition)
        if dist <= preyVicinityRange and pred.id != currentPredator.id:  # the range around the prey can be changed later when the preys are moving
            numberOfPredatorsInVicinity += 1

    if numberOfPredatorsInVicinity < 2:
        return 1/(1 + math.exp(-20 * (numberOfPredatorsInVicinity - 13) + 12))

    else:
        return 1/(1 + math.exp(20 * (numberOfPredatorsInVicinity - 16) + 13))