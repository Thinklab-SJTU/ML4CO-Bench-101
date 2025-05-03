import torch
import numpy as np
import torch.utils.data
from scipy.spatial.distance import cdist
from ml4co_kit import check_dim, to_tensor


def tsp_data_process(
    points: np.ndarray, ref_tour: np.ndarray, mode: str
):
    # check dimension
    check_dim(points, 2)
    check_dim(ref_tour, 1)
    
    # edges feature
    if mode != "solve":
        order_points = points[ref_tour]
        e = to_tensor(cdist(order_points, order_points)).float()
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)
        raw_data = order_points
    else:
        points = np.concatenate([
            points, np.expand_dims(points[0], axis=0)
        ], axis=0)
        e = to_tensor(cdist(points, points)).float()
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)
        raw_data = points
    
    return (
        None, # nodes feature, TSP does not use it
        e.float(), # (V, V, 2): edges feature, distance matrix
        raw_data
    )