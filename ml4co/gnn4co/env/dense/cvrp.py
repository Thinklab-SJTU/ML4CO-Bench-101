import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from scipy.spatial.distance import cdist
from ml4co_kit import check_dim, to_tensor


def cvrp_dense_process(
    depot: np.ndarray, points: np.ndarray, demand: np.ndarray, ref_tour: np.ndarray,
) -> Sequence[Tensor]:
    # check dimension
    check_dim(depot, 1)
    check_dim(points, 2)
    check_dim(depot, 1)
    check_dim(ref_tour, 1)

    # drop -1 for ref_tour
    if ref_tour is not None and -1 in ref_tour:
        fisrt_idx = np.where(ref_tour==-1)[0][0]
        ref_tour = ref_tour[:fisrt_idx]
    
    # nodes num
    nodes_num = len(demand) + 1
    
    # update points
    depots = depot.reshape(1, 2)
    points = np.concatenate([depots, points], 0)

    # graph
    graph = to_tensor(cdist(points, points)).float()
    e = torch.zeros_like(graph)
    
    # update demand
    demand_new = np.zeros(shape=(nodes_num,))
    demand_new[1:] = demand
    demand_new = demand_new.reshape(nodes_num, 1)
    
    # ground truth (partition matrix)
    if ref_tour is not None:   
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth + ground_truth.T
        ground_truth = torch.clip(ground_truth, 0, 1).long()
    else:
        ground_truth = None

    # node feature
    depot_flag = np.zeros(shape=(nodes_num,))
    depot_flag[0] = 1
    depot_flag = depot_flag.reshape(nodes_num, 1)
    x = np.concatenate([points, demand_new, depot_flag], 1)
    nodes_feature = to_tensor(x)
    
    # mask
    mask = torch.zeros_like(graph)

    # raw_data
    raw_data = [points, demand_new, ref_tour]
    
    return (
        nodes_feature.float(), # (V, 4): nodes feature
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth, # (V,): Ground truth
        nodes_num, # Number of nodes
    )