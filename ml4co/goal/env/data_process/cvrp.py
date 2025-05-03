import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from scipy.spatial.distance import cdist
from ml4co_kit import check_dim, to_tensor


def cvrp_data_process(
    depot: np.ndarray, points: np.ndarray, demand: np.ndarray, 
    ref_tour: np.ndarray, mode: str,
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
    
    # update demand
    demand_new = np.zeros(shape=(nodes_num + 1,))
    demand_new[1:-1] = demand
     
    if mode == "train":
        # get order [demand, remain_cp, via_depot]
        x = torch.zeros(size=(nodes_num + 1, 3))
        order = [0]
        remain_cp = 1.0
        for cur_node, next_node in zip(ref_tour[1:-1], ref_tour[2:]):
            if cur_node != 0:
                remain_cp -= demand_new[cur_node]
                order.append(cur_node)
                x[cur_node][0] = demand_new[cur_node]
                x[cur_node][1] = remain_cp
                if next_node == 0:
                    x[cur_node][2] = 1
            else:
                remain_cp = 1.0
        order.append(0)
        
        # nodes feature
        x = x[order]
        
        # edges feature
        coords = np.concatenate([np.expand_dims(depot, axis=0), points], axis=0)
        order_coords = coords[order]
        e = to_tensor(cdist(order_coords, order_coords)).float()
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)

        # raw data
        raw_data = None
        
    else:
        # nodes feature
        x = torch.zeros(size=(nodes_num + 1, 2))
        x[:, 0] = to_tensor(demand_new)
        x[0][1] = 1.0
    
        # edges feature
        coords = np.concatenate([
            np.expand_dims(depot, axis=0), points, np.expand_dims(depot, axis=0)
        ], axis=0)
        e = to_tensor(cdist(coords, coords)).float()
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)
    
        # raw data
        raw_data = [coords, demand_new]        
        
    return (
        x.float(), # nodes feature
        e.float(), # (V, V, 2): edges feature, distance matrix
        raw_data
    )