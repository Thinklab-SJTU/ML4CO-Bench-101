import numpy as np
from torch import Tensor
from ml4co_kit import to_numpy
from typing import Union, List

    
def mcut_sparse_decode(
    vars: Tensor, nodes_feature: Tensor, edges_feature: Tensor, 
    edge_index: Tensor, graph_list: List[Tensor], nodes_num_list: list, 
    edges_num_list: list, return_cost: bool
) -> Union[List[np.ndarray], np.floating]:
    vars = to_numpy(vars)
    if return_cost:
        vars = vars * 2 - 1
        edge_index = to_numpy(edge_index)
        edge_cut = vars[edge_index[0]] * vars[edge_index[1]]
        cut_number = (edge_cut == -1).sum() / 2 / len(nodes_num_list)
        return cut_number 
    else:
        solutions = list()
        begin_idx = 0
        for nodes_num in nodes_num_list:
            end_idx = begin_idx + nodes_num
            solutions.append(vars[begin_idx:end_idx])
            begin_idx = end_idx
        return solutions