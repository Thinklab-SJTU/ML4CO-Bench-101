import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import (
    to_numpy, TSPSolver
)


class DIMESDecoder(object):
    def __init__(self) -> None:
        pass
        
    def sparse_decode(
        self, heatmap: Tensor, task: str, x: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        # get solutions
        solutions = list()
        if task in ["TSP"]:
            node_begin_idx = 0
            edge_begin_idx = 0
            for idx in range(len(edges_num_list)):
                node_end_idx = node_begin_idx + edges_num_list[idx]
                edge_end_idx = edge_begin_idx + edges_num_list[idx]
                solutions.append(self._edge_sparse_decode(
                    heatmap=heatmap[edge_begin_idx:edge_end_idx], 
                    x=x[node_begin_idx:node_end_idx],
                    e=e[edge_begin_idx:edge_end_idx],
                    edge_index=edge_index[:, edge_begin_idx:edge_end_idx],
                    nodes_num=nodes_num_list[idx]
                ))
                node_begin_idx = node_end_idx
                edge_begin_idx = edge_end_idx
        else:
            raise NotImplementedError()
        
        # check if return cost
        if return_cost:
            if task in ["TSP"]:
                tsp_solver = TSPSolver()
                tsp_solver.from_data(
                    points=to_numpy(x), tours=solutions
                )
                costs = tsp_solver.evaluate()
            else:
                raise NotImplementedError()
            return costs
        else:
            return solutions
    
    def dense_decode(
        self, heatmap: Tensor, task: str, x: Tensor, graph: Tensor,
        ground_truth: Tensor, nodes_num_list: list, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        solutions = list()
        # get solutions
        if task in ["TSP"]:
            for idx in range(heatmap.shape[0]):
                solutions.append(self._edge_dense_decode(
                    heatmap=heatmap[idx], x=x[idx], graph=graph[idx]
                ))
        else:
            raise NotImplementedError()
        
        # check if return cost
        if return_cost:
            if task == "TSP":
                tsp_solver = TSPSolver()
                tsp_solver.from_data(
                    points=to_numpy(x), tours=solutions
                )
                costs = tsp_solver.evaluate()
            else:
                raise NotImplementedError()
            return costs
        else:
            return solutions
        
    def _node_sparse_decode(
        self, heatmap: Tensor, graph: Tensor
    ) -> np.ndarray:
        raise NotImplementedError(
            "``_node_sparse_decode`` is required to implemented in subclasses."
        )
        
    def _edge_sparse_decode(
        self, heatmap: Tensor, x: Tensor, e: Tensor, edge_index: Tensor, nodes_num: int
    ) -> np.ndarray:
        raise NotImplementedError(
            "``_edge_sparse_decode`` is required to implemented in subclasses."
        )
        
    def _edge_dense_decode(
        self, heatmap: Tensor, x: Tensor, graph: Tensor, 
    ) -> np.ndarray:
        raise NotImplementedError(
            "``_edge_dense_decode`` is required to implemented in subclasses."
        )