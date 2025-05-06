import numpy as np
from ml4co_kit import (
    to_numpy, to_tensor, np_sparse_to_dense, mcut_rlsa_local_search
)
from torch_geometric.utils import remove_self_loops


def mcut_decode_greedy(
    heatmap: np.ndarray, 
    num_nodes: int, 
    edge_index: np.ndarray, 
    use_rlsa: bool = True, 
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_device: str = "cuda",
    rlsa_seed: int = 1234
):
    # numpy
    heatmap = to_numpy(heatmap)

    # adj_matrix
    edge_index = to_numpy(edge_index)
    adj_matrix = np_sparse_to_dense(num_nodes, edge_index)
    adj_matrix = np.where(adj_matrix == 2, np.ones_like(adj_matrix), adj_matrix)

    # greedy decoding
    sol = (heatmap > 0.5).astype(np.int32)

    # local search
    if use_rlsa:
        sol = mcut_rlsa_local_search(
            init_sol=sol,
            graph=adj_matrix,
            edge_index=edge_index,
            rlsa_kth_dim="both",
            rlsa_tau=rlsa_tau,
            rlsa_d=rlsa_d,
            rlsa_k=rlsa_k,
            rlsa_t=rlsa_t,
            rlsa_device=rlsa_device,
            seed=rlsa_seed
        )
    
    return to_tensor(sol), np.sum(sol).astype(np.int32)