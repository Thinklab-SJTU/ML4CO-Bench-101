import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import (
    to_numpy, MCutSolver, ATSPSolver, TSPSolver, CVRPSolver
)


class GNN4CODecoder(object):
    def __init__(self) -> None:
        pass
        
    def sparse_decode(
        self, heatmap: Tensor, task: str, x: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        # get solutions
        solutions = list()
        if task in ["MIS", "MVC", "MCl"]:
            begin_idx = 0
            for idx in range(len(graph_list)):
                end_idx = begin_idx + nodes_num_list[idx]
                solutions.append(self._node_sparse_decode(
                    heatmap=heatmap[begin_idx:end_idx], 
                    graph=graph_list[idx]
                ))
                begin_idx = end_idx
        elif task in ["MCut"]:
            node_begin_idx = 0
            edge_begin_idx = 0
            for idx in range(len(edges_num_list)):
                node_end_idx = node_begin_idx + edges_num_list[idx]
                edge_end_idx = edge_begin_idx + edges_num_list[idx]
                solutions.append(self._node_sparse_decode(
                    heatmap=heatmap[node_begin_idx:node_end_idx],
                    graph=graph_list[idx], 
                    edge_index=edge_index[:, edge_begin_idx:edge_end_idx],
                ))
                node_begin_idx = node_end_idx
                edge_begin_idx = edge_end_idx
        elif task in ["TSP"]:
            node_begin_idx = 0
            edge_begin_idx = 0
            for idx in range(len(edges_num_list)):
                node_end_idx = node_begin_idx + edges_num_list[idx]
                edge_end_idx = edge_begin_idx + edges_num_list[idx]
                solutions.append(self._edge_sparse_decode(
                    heatmap=heatmap[edge_begin_idx:edge_end_idx], 
                    x=x[node_begin_idx:node_end_idx],
                    edge_index=edge_index[:, edge_begin_idx:edge_end_idx],
                    nodes_num=nodes_num_list[idx]
                ))
                node_begin_idx = node_end_idx
                edge_begin_idx = edge_end_idx
        else:
            raise NotImplementedError()
        
        # check if return cost
        if return_cost:
            if task in ["MIS", "MVC", "MCl"]:
                costs = [sum(sol) for sol in solutions]
                costs = np.average(np.array(costs))
            elif task in ["MCut"]:
                edge_index = to_numpy(edge_index)
                mcut_solver = MCutSolver()
                mcut_solver.from_adj_matrix(
                    adj_matrix=[to_numpy(g) for g in graph_list],
                    nodes_label=solutions
                )
                costs = mcut_solver.evaluate()
            elif task in ["TSP"]:
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
        if task in ["ATSP", "TSP", "CVRP"]:
            for idx in range(heatmap.shape[0]):
                solutions.append(self._edge_dense_decode(
                    heatmap=heatmap[idx], x=x[idx], graph=graph[idx]
                ))
        else:
            raise NotImplementedError()
        
        # check if return cost
        if return_cost:
            if task == "ATSP":
                atsp_solver = ATSPSolver()
                atsp_solver.from_data(
                    dists=to_numpy(graph), tours=solutions
                )
                costs = atsp_solver.evaluate()
            elif task == "TSP":
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
        
    def _node_sparse_decode(self) -> np.ndarray:
        raise NotImplementedError(
            "``_node_sparse_decode`` is required to implemented in subclasses."
        )
        
    def _edge_sparse_decode(self) -> np.ndarray:
        raise NotImplementedError(
            "``_edge_sparse_decode`` is required to implemented in subclasses."
        )
        
    def _edge_dense_decode(self) -> np.ndarray:
        raise NotImplementedError(
            "``_edge_dense_decode`` is required to implemented in subclasses."
        )