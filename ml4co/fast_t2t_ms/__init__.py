# Prefer conda libstdc++ before Ascend / system libs (must run first).
# Importing ms_utils triggers LD_LIBRARY_PATH fix (+ re-exec if needed).
from ml4co.ms_utils import setup_ld_library_path, skip_ortools

setup_ld_library_path()
skip_ortools()

from .tsp import *
from .solver import *
