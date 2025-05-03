import torch
import numpy as np
import torch.utils.data
from scipy.spatial.distance import cdist
from ml4co_kit import check_dim, to_tensor


def tsp_dense_process(points: np.ndarray, ref_tour: np.ndarray):
    # check dimension
    check_dim(points, 2)
    check_dim(ref_tour, 1)

    # nodes_num
    nodes_num = points.shape[0]
    
    # x and graph
    x = to_tensor(points)
    graph = to_tensor(cdist(points, points)).float()
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth + ground_truth.T
    else:
        ground_truth = None

    return (
        x, # (V, 2): nodes feature, random init
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
    )