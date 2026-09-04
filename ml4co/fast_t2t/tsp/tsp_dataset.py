import torch
import numpy as np
from sklearn.neighbors import KDTree
from ml4co_kit import TSPWrapper, TSPTask, to_tensor
from ml4co.fast_t2t.common import MetaDataset, MetaData


def tsp_data_process(task_data: TSPTask) -> MetaData:
    # Get original data
    points = task_data.points
    ref_tour = task_data.ref_sol
    nodes_num: int = task_data.nodes_num

    # KNN
    knn = 50
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    dist_knn, idx_knn = kdt.query(points, k=knn, return_distance=True)
    dist_knn: np.ndarray

    # Edge index (2, E)
    edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
    edge_index_0 = edge_index_0.repeat(1, knn).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
    edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

    # Node feature (V, 2)
    node_feature = to_tensor(points).float()

    # Edge feature (E, 1)
    edge_feature = to_tensor(dist_knn.reshape(-1))

    # Ground truth (E,)
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int64)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, knn).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        
        tour_edges_rv = np.zeros(nodes_num, dtype=np.int64)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv = torch.from_numpy(tour_edges_rv)
        tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, knn).reshape(-1)
        tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)
        ground_truth = (tour_edges + tour_edges_rv).reshape(-1).long()
    else:
        ground_truth = None

    # Create MetaData
    return MetaData(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        ground_truth=ground_truth
    )


class TSPDataset(MetaDataset):
    def __init__(
        self,
        num_per_epoch: int,
        data_path: str = None,
        data_folder: str = None,
    ):
        # Super Args
        super(TSPDataset, self).__init__(
            wrapper=TSPWrapper(),
            num_per_epoch=num_per_epoch,
            data_path=data_path,
            data_folder=data_folder,
        )
        
    def data_process(self, task_data: TSPTask) -> MetaData:
        return tsp_data_process(task_data=task_data)
