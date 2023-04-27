# Using enum class create enumerations
from enum import Enum

class TeamCompositions(Enum):
   ALL_POWER = "ALL_POWER"
   ALL_AFF = "ALL_AFF"
   ALL_ACH = "ALL_ACH"
   MIXED = "MIXED" #
   AFF_POW_ACH = "AFF_POW_ACH"
   AFF_POW = "AFF_POW"
   AFF_ACH = "AFF_ACH"
   POW_ACH = "POW_ACH"

class MotiveProfiles(Enum):
    POWER = "power"
    AFFILIATION = "affiliation"
    ACHIEVEMENT = "achievement"

