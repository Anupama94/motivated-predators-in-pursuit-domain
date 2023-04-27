from numba import int32, float32, types
from numba.experimental import jitclass
import numpy as np

preySpec = [
    ('id', int32),
    ('currentPosition', int32[:]),
    ('type', types.unicode_type),
    ('status', types.unicode_type),
    ('sig', int32),
    ('stamina', float32),

]

@jitclass(preySpec)
class Prey:
    def __init__(self, id):
        self.id = id
        self.type = "PREY"
        self.status = "ALIVE"
        self.sig = 0
        # self.numberOfUnoccupiedNeighbours = 0
        self.stamina = 0.2


predatorSpec = [
    ('id', int32),          # an array field
    ('currentPosition', int32[:]),
    ('type', types.unicode_type),
    ('policy', types.unicode_type),
    ('targetPrey', Prey.class_type.instance_type),
    ('targetDestination', int32[:]),

]

@jitclass(predatorSpec)
class Predator:
    def __init__(self, id):
        self.id = id
        self.type = "PREDATOR"
        self.policy = ""
        # self.targetPrey = None
        # self.targetDestination = np.array([0, 0])

