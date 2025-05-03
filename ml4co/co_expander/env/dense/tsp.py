import torch
import numpy as np
from scipy.spatial.distance import cdist
from ml4co_kit import check_dim, to_tensor


def tsp_dense_process(points: np.ndarray, ref_tour: np.ndarray):
    # check dimension
    check_dim(points, 2)
    check_dim(ref_tour, 1)

    # nodes_num
    nodes_num = points.shape[0]
    
    # graph
    graph = to_tensor(cdist(points, points)).float()
    e = torch.zeros_like(graph)
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth + ground_truth.T
        ground_truth = ground_truth.long()
    else:
        ground_truth = None
    
    # mask
    nodes_feature = to_tensor(points)
    mask = torch.zeros_like(graph)

    # raw_data
    raw_data = [points, ref_tour]

    return (
        nodes_feature.float(), # (V, 2): nodes' feature, (x, y) euler coordinates of nodes
        None, # nodes' decision variable, not used for TSP
        graph.float(), # (V, V): edges' feature, for TSP, it's distance matrix
        e.float(), # (V, V): edges' decision variable, whether to select an edge or not 
        mask.bool(), # (V,): mask
        ground_truth, # (V,): ground truth
        nodes_num, # number of nodes
        raw_data, # original data
    )