import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from sklearn.neighbors import KDTree
from ml4co_kit import check_dim, to_tensor


def tsp_sparse_process(
    points: np.ndarray, ref_tour: np.ndarray, sparse_factor: int
) -> Sequence[Tensor]:
    # check dimension
    check_dim(points, 2)
    check_dim(ref_tour, 1)

    # nodes_num and edges_num
    nodes_num = points.shape[0]
    edges_num = nodes_num * sparse_factor
    
    # KDTree        
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    dists_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)
    edges_feature = to_tensor(dists_knn.reshape(-1))
    
    # edge_index
    edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
    edge_index_0 = edge_index_0.repeat(1, sparse_factor).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
    edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
    
    # ground truth
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int64)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        
        tour_edges_rv = np.zeros(nodes_num, dtype=np.int64)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv = torch.from_numpy(tour_edges_rv)
        tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)
        ground_truth = (tour_edges + tour_edges_rv).reshape(-1).long()
    else:
        ground_truth = None
    
    # other variables
    nodes_feature = to_tensor(points)
    e = torch.zeros_like(edges_feature)
    mask = torch.zeros_like(edges_feature)
    
    # raw_data
    raw_data = [points, ref_tour]
    
    return (
        nodes_feature.float(), # (V, 2): nodes' feature, (x, y) euler coordinates of nodes
        None, # (V,): nodes' decision variable, not used for TSP
        edges_feature.float(), # (E,): edges' feature, for TSP, it's distance between nodes
        e.float(), # (E,): edges' decision variable, whether to select an edge or not
        edge_index.long(), # (2, E): index of edge endpoints
        None, # adjacency matrix, not used for TSP
        mask.bool(), # (V,): mask
        ground_truth, # (V,): ground truth
        nodes_num, # number of nodes
        edges_num, # number of edges
        raw_data, # original data
    )