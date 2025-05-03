import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import to_numpy, mcut_rlsa_local_search

    
def mcut_sparse_decode(
    vars: Tensor, 
    edge_index: Tensor, 
    graph_list: List[Tensor], 
    nodes_num_list: list, 
    edges_num_list: list, 
    return_cost: bool,
    use_rlsa: bool = True, 
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_device: str = "cuda",
    rlsa_seed: int = 1234
) -> Union[List[np.ndarray], np.floating]:
    vars = to_numpy(vars)
    edge_index = to_numpy(edge_index)
    if return_cost:
        vars = vars * 2 - 1
        edge_cut = vars[edge_index[0]] * vars[edge_index[1]]
        cut_number = (edge_cut == -1).sum() / 2 / len(nodes_num_list)
        return cut_number 
    else:
        if use_rlsa == False:
            solutions = list()
            begin_idx = 0
            for nodes_num in nodes_num_list:
                end_idx = begin_idx + nodes_num
                solutions.append(vars[begin_idx:end_idx])
                begin_idx = end_idx
            return solutions
        else:
            solutions = list()
            node_begin_idx = 0
            edge_begin_idx = 0
            for graph, nodes_num, edges_num in zip(graph_list, nodes_num_list, edges_num_list):
                node_end_idx = node_begin_idx + nodes_num
                edge_end_idx = edge_begin_idx + edges_num
                solutions.append(
                    mcut_rlsa_local_search(
                        init_sol=vars[node_begin_idx:node_end_idx],
                        graph=to_numpy(graph),
                        edge_index=edge_index[edge_begin_idx:edge_end_idx],
                        rlsa_tau=rlsa_tau,
                        rlsa_d=rlsa_d,
                        rlsa_k=rlsa_k,
                        rlsa_t=rlsa_t,
                        rlsa_device=rlsa_device,
                        seed=rlsa_seed
                    )
                )
                node_begin_idx = node_end_idx
                edge_begin_idx = edge_end_idx
            return solutions