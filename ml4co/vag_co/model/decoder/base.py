import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import to_numpy, MCutSolver


class VAGCODecoder(object):
    def __init__(self) -> None:
        pass
        
    def decode(
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
                solutions.append(self._decode(
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
                solutions.append(self._decode(
                    heatmap=heatmap[node_begin_idx:node_end_idx],
                    graph=graph_list[idx], 
                    edge_index=edge_index[:, edge_begin_idx:edge_end_idx],
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
            else:
                raise NotImplementedError()
            return costs
        else:
            return solutions
     
    def _decode(self) -> np.ndarray:
        raise NotImplementedError(
            "``_decode`` is required to implemented in subclasses."
        )
        
  