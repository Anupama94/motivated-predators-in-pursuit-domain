
from numba import types
from numba import int32, boolean
from numba.experimental import jitclass



spec = [
    ('stationaryPreys', int32),               # a simple scalar field
    ('assign_agent_locations', types.unicode_type),
    ('grid_size', int32),
    ('prey_count', int32),
    ('predator_count', int32),
    ('prey_capture', int32),
    ('enable_gui', boolean),
    ('regiment', types.unicode_type),
    ('render_frequency', types.float32),
    ('motive_profile_ratio', int32[:]),
    ('significance_weight', types.float64),
    ('adjacent_weight', types.float64),
    ('threshold_distance', types.int32),
    ('local_aff_range', types.int32),
    ('global_aff_range', types.int32),
    ('aff_count', types.int32),
    ('pow_count', types.int32),
    ('ach_count', types.int32),
    ('difficulty_level', types.int32),

]



@jitclass(spec)
class ArgumentList:

    def __init__(self):
        self.stationaryPreys = 1
        self.assign_agent_locations = "RANDOM"
        self.grid_size = 16
        self.prey_count = 12
        self.predator_count = 12
        self.prey_capture = 4
        self.enable_gui = True
        self.regiment = "ALL_POWER"
        self.render_frequency = 0.2
        self.difficulty_level = 0
        self.significance_weight = 0.3
        self.adjacent_weight = 0.7
        self.threshold_distance = 5
        self.local_aff_range = 3
        self.global_aff_range = 1





