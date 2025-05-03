from .base import COExpanderEmbedder
from .atsp import ATSPEmbedder
from .cvrp import CVRPEmbedder
from .mcl import MClEmbedder
from .mcut import MCutEmbedder
from .mis import MISEmbedder
from .mvc import MVCEmbedder
from .tsp import TSPEmbedder
from .time import timestep_embedding


EMBEDDER_DICT = {
    "ATSP": ATSPEmbedder,
    "CVRP": CVRPEmbedder,
    "MCl": MClEmbedder,
    "MIS": MISEmbedder,
    "MCut": MCutEmbedder,
    "MVC": MVCEmbedder,
    "TSP": TSPEmbedder
}

def get_embedder_by_task(task: str):
    return EMBEDDER_DICT[task]