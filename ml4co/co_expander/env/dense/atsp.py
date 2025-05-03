import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import check_dim, to_tensor


def atsp_dense_process(dists: np.ndarray, ref_tour: np.ndarray) -> Sequence[Tensor]:
    # check dimension
    check_dim(dists, 2)
    check_dim(ref_tour, 1)

    # nodes_num
    nodes_num = dists.shape[0]

    # graph
    graph = to_tensor(dists).float()
    e = torch.zeros_like(graph)
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth.long()
    else:
        ground_truth = None

    # mask
    mask = torch.zeros_like(graph)

    # raw_data
    raw_data = [dists, ref_tour]
    
    return (
        None, # nodes' feature, not used for ATSP
        None, # nodes' decision variable, not used for ATSP
        graph.float(), # (V, V): edges' feature, it's cost matrix
        e.float(), # (V, V): nodes' decision variable, whether to select an edge or not 
        mask.bool(), # (V,): mask
        ground_truth, # (V,): ground truth
        nodes_num, # number of nodes
        raw_data, # original data
    )