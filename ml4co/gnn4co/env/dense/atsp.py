import torch
import numpy as np
import torch.utils.data
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
    x = torch.randn(size=(nodes_num, 2))
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
    else:
        ground_truth = None
    
    return (
        x, # (V, 2): nodes feature, random init
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
    )