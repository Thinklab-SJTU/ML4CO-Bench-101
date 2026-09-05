import numpy as np
import mindspore as ms
from sklearn.neighbors import KDTree
from ml4co_kit import TSPWrapper, TSPTask
from ml4co.fast_t2t_ms.common import MetaDataset, MetaData


def tsp_data_process(task_data: TSPTask) -> MetaData:
    """
    Build sparse KNN graph features on **host (NumPy)**.

    Do NOT use ``ops.*`` / Ascend here: small Ascend kernels for preprocessing
    dominate wall time (~1s/instance in diagnose) while the net forward is ~10ms.
    Device placement happens once later via ``MetaDataBatch.to_device``.
    """
    points = np.asarray(task_data.points, dtype=np.float32)
    ref_tour = task_data.ref_sol
    nodes_num: int = task_data.nodes_num

    # KNN (CPU)
    knn = 50
    kdt = KDTree(points, leaf_size=30, metric="euclidean")
    dist_knn, idx_knn = kdt.query(points, k=knn, return_distance=True)

    # Edge index (2, E)
    edge_index_0 = np.repeat(np.arange(nodes_num, dtype=np.int32), knn)
    edge_index_1 = idx_knn.reshape(-1).astype(np.int32)
    edge_index = np.stack([edge_index_0, edge_index_1], axis=0)

    # Node / edge features
    node_feature = points.astype(np.float32, copy=False)
    edge_feature = dist_knn.reshape(-1).astype(np.float32)

    # Ground truth (E,): undirected tour edges aligned with KNN slots
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int32)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = np.repeat(tour_edges, knn)
        tour_edges_mask = edge_index_1 == tour_edges

        tour_edges_rv = np.zeros(nodes_num, dtype=np.int32)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv = np.repeat(tour_edges_rv, knn)
        tour_edges_rv_mask = edge_index_1 == tour_edges_rv
        ground_truth = (tour_edges_mask | tour_edges_rv_mask).astype(np.int32)
    else:
        ground_truth = None

    # Host MindSpore Tensors (CPU). Moved to Ascend in batch.to_device().
    return MetaData(
        node_feature=ms.Tensor(node_feature),
        edge_feature=ms.Tensor(edge_feature),
        edge_index=ms.Tensor(edge_index),
        ground_truth=(
            None if ground_truth is None else ms.Tensor(ground_truth)
        ),
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
