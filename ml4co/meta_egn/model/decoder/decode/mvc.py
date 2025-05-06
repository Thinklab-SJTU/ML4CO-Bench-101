import numpy as np
from ml4co_kit import (
    to_numpy, to_tensor, np_sparse_to_dense, 
    mvc_greedy_decoder, mvc_rlsa_local_search
)


def mvc_decode_greedy(
    heatmap: np.ndarray, 
    num_nodes: int, 
    edge_index: np.ndarray, 
    use_rlsa: bool = True, 
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_beta: float = 1.02, 
    rlsa_alpha: float = 0.3, 
    rlsa_device: str = "cuda",
    rlsa_seed: int = 1234
):
    # numpy
    heatmap = to_numpy(heatmap)

    # adj_matrix
    edge_index = to_numpy(edge_index)
    adj_matrix = np_sparse_to_dense(num_nodes, edge_index)
    adj_matrix = np.clip(adj_matrix, a_min=0, a_max=1)

    # greedy decoding
    sol = mvc_greedy_decoder(heatmap, adj_matrix)

    # local search
    if use_rlsa:
        sol = mvc_rlsa_local_search(
            init_sol=sol,
            graph=adj_matrix,
            rlsa_kth_dim="both",
            rlsa_tau=rlsa_tau,
            rlsa_d=rlsa_d,
            rlsa_k=rlsa_k,
            rlsa_t=rlsa_t,
            rlsa_alpha=rlsa_alpha,
            rlsa_beta=rlsa_beta,
            rlsa_device=rlsa_device,
            seed=rlsa_seed
        )
    
    return to_tensor(sol), np.sum(sol).astype(np.int32)