import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import to_numpy, mcl_rlsa_local_search


def mcl_sparse_decode(
    vars: Tensor, 
    graph_list: List[Tensor], 
    nodes_num_list: list, 
    return_cost: bool, 
    use_rlsa: bool = True, 
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_beta: float = 1.02, 
    rlsa_alpha: float = 0.3, 
    rlsa_device: str = "cuda",
    rlsa_seed: int = 1234
) -> Union[List[np.ndarray], np.floating]:
    vars = to_numpy(vars)
    if return_cost:
        return np.sum(vars) / len(nodes_num_list)
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
            begin_idx = 0
            for graph, nodes_num in zip(graph_list, nodes_num_list):
                end_idx = begin_idx + nodes_num
                solutions.append(
                    mcl_rlsa_local_search(
                        init_sol=vars[begin_idx:end_idx],
                        graph=to_numpy(graph),
                        rlsa_tau=rlsa_tau,
                        rlsa_d=rlsa_d,
                        rlsa_k=rlsa_k,
                        rlsa_t=rlsa_t,
                        rlsa_alpha=rlsa_alpha,
                        rlsa_beta=rlsa_beta,
                        rlsa_device=rlsa_device,
                        seed=rlsa_seed
                    )
                )
                begin_idx = end_idx
            return solutions