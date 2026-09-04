# Avoid ml4co_kit pulling in ortools (libre2 / protobuf conflicts on Ascend).
from .skip_ortools import apply as _skip_ortools

_skip_ortools()

from .tsp import *
from .solver import *
