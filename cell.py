
from numba import int32, types
from numba.experimental import jitclass

PREDATOR = "PREDATOR"
PREY = "PREY"

# # key and value types
# kv_ty = (types.int64, types.unicode_type)

spec = [
    ('row', int32),
    ('col', int32),

    ('predatorId', types.unicode_type),          # an array field
    ('preyId', types.unicode_type),
]

@jitclass(spec)
class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.predatorId = "0"
        self.preyId = "0"


nodespec = [
    ('currentCell', int32),
    ('parentCell', int32),
    ('gScore', int32),
    ('hScore', int32),
    ('fScore', int32),
]

@jitclass(nodespec)
class Node:
    def __init__(self, currentCell, parentCell, gScore, hScore, fScore):
        self.currentCell = currentCell
        self.parentCell = parentCell
        self.gScore = gScore
        self.hScore = hScore
        self.fScore = fScore


