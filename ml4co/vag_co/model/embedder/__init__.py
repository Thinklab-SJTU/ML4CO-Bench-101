from .base import GNN4COEmbedder
from .mcl import MClEmbedder
from .mcut import MCutEmbedder
from .mis import MISEmbedder
from .mvc import MVCEmbedder


EMBEDDER_DICT = {
    "MCl": MClEmbedder,
    "MIS": MISEmbedder,
    "MCut": MCutEmbedder,
    "MVC": MVCEmbedder,
}

def get_embedder_by_task(task: str):
    return EMBEDDER_DICT[task]