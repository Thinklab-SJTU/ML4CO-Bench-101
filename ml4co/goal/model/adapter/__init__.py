from .base import GOALNodeAdapter, GOALEdgeAdapter, GOALOutputAdapter
from .atsp import ATSPNodeAdapter, ATSPEdgeAdapter, ATSPOutputAdapter
from .tsp import TSPNodeAdapter, TSPEdgeAdapter, TSPOutputAdapter
from .cvrp import CVRPNodeAdapter, CVRPEdgeAdapter, CVRPOutputAdapter
from .mis import MISNodeAdapter, MISEdgeAdapter, MISOutputAdapter
from .mcl import MClNodeAdapter, MClEdgeAdapter, MClOutputAdapter
from .mvc import MVCNodeAdapter, MVCEdgeAdapter, MVCOutputAdapter


NODE_ADAPTER_DICT = {
    "ATSP": ATSPNodeAdapter,
    "TSP": TSPNodeAdapter,
    "CVRP": CVRPNodeAdapter,
    "MIS": MISNodeAdapter,
    "MCl": MClNodeAdapter,
    "MVC": MVCNodeAdapter,
}


EDGE_ADAPTER_DICT = {
    "ATSP": ATSPEdgeAdapter,
    "TSP": TSPEdgeAdapter,
    "CVRP": CVRPEdgeAdapter,
    "MIS": MISEdgeAdapter,
    "MCl": MClEdgeAdapter,
    "MVC": MVCEdgeAdapter,
}


OUTPUT_ADAPTER_DICT = {
    "ATSP": ATSPOutputAdapter,
    "TSP": TSPOutputAdapter,
    "CVRP": CVRPOutputAdapter,
    "MIS": MISOutputAdapter,
    "MCl": MClOutputAdapter,
    "MVC": MVCOutputAdapter,
}


def get_node_adapter_by_task(task: str):
    return NODE_ADAPTER_DICT[task]


def get_edge_adapter_by_task(task: str):
    return EDGE_ADAPTER_DICT[task]


def get_output_adapter_by_task(task: str):
    return OUTPUT_ADAPTER_DICT[task]