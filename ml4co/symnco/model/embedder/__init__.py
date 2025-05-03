from .base import SymNCODynamicEmbedder
from .tsp import TSPInitEmbedder, TSPContextEmbedder
from .cvrp import CVRPInitEmbedder, CVRPContextEmbedder


INIT_EMBEDDER_DICT = {
    "TSP": TSPInitEmbedder,
    "CVRP": CVRPInitEmbedder
}
CONTEXT_EMBEDDER_DICT = {
    "TSP": TSPContextEmbedder,
    "CVRP": CVRPContextEmbedder
}


def get_init_embedder_by_task(task: str):
    return INIT_EMBEDDER_DICT[task]


def get_context_embedder_by_task(task: str):
    return CONTEXT_EMBEDDER_DICT[task]