import numpy as np

from numba import int32, float32, deferred_type, njit
from numba.experimental import jitclass
from numba.core import types
from numba.typed import List

from grid import Grid
from agent import Prey, Predator

@njit
def delete_workaround(arr, num):
    mask = np.zeros(arr.shape[0], dtype=Prey.class_type.instance_type) == 0
    mask[np.where(arr == num)[0]] = False
    return arr[mask]

spec = [
    ('arena', Grid.class_type.instance_type),               # a simple scalar field
    ('capturedState', int32),
    ('predators', types.ListType(Predator.class_type.instance_type)),
    ('numberOfPreysLeft', int32),
    ('preys', types.ListType(Prey.class_type.instance_type))

]



@jitclass(spec)
class Game:
    def __init__(self, grid, capturedState, preys, predators):
        self.arena = grid
        self.capturedState = capturedState # how many predators it takes to capture a prey
        self.preys = preys
        self.predators = predators
        self.numberOfPreysLeft = len(preys)

    def isPreyCaptured(self, prey):
        # print("checking")
        vacantPositionsAroundPrey = self.arena.getUnoccupiedNeighbourCellCount(prey.currentPosition)
        if vacantPositionsAroundPrey == 0:
            # check whether the prey is surrounded by at least 1 predator
            if self.arena.getUnoccupiedNeighbourCellCount(prey.currentPosition, "PREDATOR") <= 3:
                return True
            return False
        return False

    def isGameOver(self):
        return self.numberOfPreysLeft == 0
