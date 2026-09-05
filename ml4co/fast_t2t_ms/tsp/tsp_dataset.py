import numpy as np
import mindspore as ms
from mindspore import ops
from sklearn.neighbors import KDTree
from ml4co_kit import TSPWrapper, TSPTask
from ml4co.ms_utils import current_ms_device
from ml4co.ms_utils.type_utils import to_tensor
from ml4co.fast_t2t_ms.common import MetaDataset, MetaData


def tsp_data_process(task_data: TSPTask) -> MetaData:
    # Get original data
    points = task_data.points
    ref_tour = task_data.ref_sol
    nodes_num: int = task_data.nodes_num
    device = current_ms_device()

    # KNN
    knn = 50
    kdt = KDTree(points, leaf_size=30, metric="euclidean")
    dist_knn, idx_knn = kdt.query(points, k=knn, return_distance=True)

    # Edge index (2, E) — host numpy → move_to(device); bare Tensor stays CPU
    edge_index_0 = ops.arange(0, nodes_num, dtype=ms.int32).reshape((-1, 1))
    edge_index_0 = ops.broadcast_to(edge_index_0, (nodes_num, knn)).reshape(-1)
    edge_index_1 = to_tensor(
        idx_knn.reshape(-1).astype(np.int32), ms.int32, device=device
    )
    edge_index = ops.stack([edge_index_0, edge_index_1], axis=0)

    # Node feature (V, 2)
    node_feature = to_tensor(points, ms.float32, device=device)

    # Edge feature (E,)
    edge_feature = to_tensor(
        dist_knn.reshape(-1).astype(np.float32), ms.float32, device=device
    )

    # Ground truth (E,): undirected tour edges aligned with KNN slots
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int32)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges_t = to_tensor(tour_edges, ms.int32, device=device)
        tour_edges_t = ops.broadcast_to(
            tour_edges_t.reshape((-1, 1)), (nodes_num, knn)
        ).reshape(-1)
        tour_edges_mask = ops.equal(edge_index_1, tour_edges_t).reshape(-1, 1)

        tour_edges_rv = np.zeros(nodes_num, dtype=np.int32)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv_t = to_tensor(tour_edges_rv, ms.int32, device=device)
        tour_edges_rv_t = ops.broadcast_to(
            tour_edges_rv_t.reshape((-1, 1)), (nodes_num, knn)
        ).reshape(-1)
        tour_edges_rv_mask = ops.equal(edge_index_1, tour_edges_rv_t).reshape(
            -1, 1
        )
        ground_truth = (tour_edges_mask + tour_edges_rv_mask).reshape(-1).astype(
            ms.int32
        )
    else:
        ground_truth = None

    # Create MetaData
    return MetaData(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        ground_truth=ground_truth,
    )


class TSPDataset(MetaDataset):
    def __init__(
        self,
        num_per_epoch: int,
        data_path: str = None,
        data_folder: str = None,
    ):
        # Super Args
        super().__init__(
            wrapper=TSPWrapper(),
            num_per_epoch=num_per_epoch,
            data_path=data_path,
            data_folder=data_folder,
        )

    def data_process(self, task_data: TSPTask) -> MetaData:
        return tsp_data_process(task_data=task_data)
