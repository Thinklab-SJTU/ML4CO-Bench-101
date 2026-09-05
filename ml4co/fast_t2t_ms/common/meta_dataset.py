import os
from typing import List, Tuple
from mindspore import Tensor
import mindspore as ms
import numpy as np
from ml4co_kit import WrapperBase, TaskBase
from ml4co_kit.learning.extra_backends.mindspore import MSDataset


class MetaData(object):
    def __init__(
        self,
        node_feature: Tensor,
        edge_feature: Tensor,
        edge_index: Tensor,
        ground_truth: Tensor,
    ):
        """
        Args:
            node_feature: Node features (V, D_v)
                - The feature of each node
            edge_feature: Edge features (E, D_e)
                - The feature of each edge
            edge_index: Edge indices (2, E)
                - The indices of the start and end nodes of each edge
            ground_truth: Ground truth (V,) or (E,)
                - The ground truth of the graph
        """
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.edge_index = edge_index
        self.ground_truth = ground_truth


class MetaDataBatch(object):
    """
    Batched sparse graph data, implemented with pure MindSpore
    (replacement for torch_geometric.data.Batch / the PyTorch MetaDataBatch).
    """

    def __init__(
        self,
        node_feature: Tensor = None,
        edge_feature: Tensor = None,
        edge_index: Tensor = None,
        ground_truth: Tensor = None,
        batch: Tensor = None,
        ptr: Tensor = None,
    ):
        # Initialize attributes
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.edge_index = edge_index
        self.ground_truth = ground_truth
        self.batch = batch
        self.ptr = ptr

    def from_data_list(self, data_list: List[MetaData]):
        """
        Combine a list of MetaData into this MetaDataBatch by concatenating
        nodes/edges and offsetting edge_index, matching PyG Batch semantics.

        Concatenation is done on **host NumPy** to avoid Ascend ``ops.cat`` /
        ``ops.full`` launch overhead during solve preprocessing.
        """
        from ml4co.ms_utils.type_utils import to_numpy

        node_features = []
        edge_features = []
        edge_indices = []
        ground_truths = []
        batch_vecs = []
        ptr = [0]
        node_offset = 0
        has_gt = data_list[0].ground_truth is not None

        for i, data in enumerate(data_list):
            nf = to_numpy(data.node_feature)
            ef = to_numpy(data.edge_feature)
            ei = to_numpy(data.edge_index).astype(np.int32, copy=False)
            num_nodes = int(nf.shape[0])

            node_features.append(nf)
            edge_features.append(ef)
            edge_indices.append(ei + node_offset)
            if has_gt:
                ground_truths.append(to_numpy(data.ground_truth))
            batch_vecs.append(np.full((num_nodes,), i, dtype=np.int32))
            node_offset += num_nodes
            ptr.append(node_offset)

        # Host tensors; ``to_device`` moves them once onto Ascend/GPU/CPU.
        self.node_feature = ms.Tensor(
            np.concatenate(node_features, axis=0).astype(np.float32, copy=False)
        )
        self.edge_feature = ms.Tensor(
            np.concatenate(edge_features, axis=0).astype(np.float32, copy=False)
        )
        self.edge_index = ms.Tensor(np.concatenate(edge_indices, axis=1))
        if has_gt:
            self.ground_truth = ms.Tensor(
                np.concatenate(ground_truths, axis=0).astype(np.int32, copy=False)
            )
        else:
            self.ground_truth = None
        self.batch = ms.Tensor(np.concatenate(batch_vecs, axis=0))
        self.ptr = ms.Tensor(np.asarray(ptr, dtype=np.int32))


    def to_device(self, device: str = None):
        """
        Move batch tensors onto ``device``.

        Skips ``move_to`` when a tensor is already on the target device
        (repeated Ascend ``move_to`` is expensive).
        """
        from ml4co.ms_utils import normalize_ms_device, maybe_move_tensor

        if device is not None:
            device = normalize_ms_device(device)
        # strict=True: silent CPU leftover makes Ascend look extremely slow.
        self.node_feature = maybe_move_tensor(
            self.node_feature, device, strict=True
        )
        self.edge_feature = maybe_move_tensor(
            self.edge_feature, device, strict=True
        )
        self.edge_index = maybe_move_tensor(
            self.edge_index, device, strict=True
        )
        self.ground_truth = maybe_move_tensor(
            self.ground_truth, device, strict=True
        )
        self.batch = maybe_move_tensor(self.batch, device, strict=True)
        self.ptr = maybe_move_tensor(self.ptr, device, strict=True)

    # Alias used by PyTorch code paths
    to_cuda = to_device

    @staticmethod
    def _maybe_move(tensor: Tensor, device: str = None) -> Tensor:
        from ml4co.ms_utils import maybe_move_tensor
        return maybe_move_tensor(tensor, device)


class MetaDataset(MSDataset):
    def __init__(
        self,
        wrapper: WrapperBase,
        num_per_epoch: int,
        data_path: str = None,
        data_folder: str = None,
    ):
        # Super Args
        super().__init__()

        # Set Attributes (Data related)
        self.wrapper = wrapper
        self.num_per_epoch = num_per_epoch
        self.data_folder = data_folder

        # Data list
        self.data_list = list()

        # Process data path
        if data_folder is not None:
            self.sub_files = [
                os.path.join(self.data_folder, file)
                for file in os.listdir(self.data_folder)
            ]
        else:
            self.sub_files = [data_path]
        self.sub_files_num = len(self.sub_files)
        self.left_num = 0

    def load_data(self, data_path: str):
        # Log
        print(f"Loading data from {data_path}")

        # Load data
        if data_path.endswith(".pkl"):
            self.wrapper.from_pickle(data_path)
        elif data_path.endswith(".txt"):
            self.wrapper.from_txt(data_path, ref=True)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # Update data list and shuffle
        self.data_list = self.wrapper.task_list
        self.left_num = len(self.data_list)
        np.random.shuffle(self.data_list)

    def __len__(self):
        return self.num_per_epoch

    def __getitem__(self, idx: int) -> Tuple[TaskBase, MetaData]:
        # Check if the data is empty
        if self.left_num == 0:
            sel_idx = np.random.randint(0, self.sub_files_num, size=(1,))[0]
            data_path: str = self.sub_files[sel_idx]
            self.load_data(data_path)

        # Update left number
        self.left_num -= 1

        # Return the meta data
        task_data = self.data_list[self.left_num - 1]
        meta_data = self.data_process(task_data)
        return task_data, meta_data

    def collate_fn(self, batch: List[Tuple[TaskBase, MetaData]]) -> MetaDataBatch:
        # Process batch data
        batch_task_data = list()
        processed_data_list = list()
        for (task_data, meta_data) in batch:
            batch_task_data.append(task_data)
            processed_data_list.append(meta_data)

        # Create MetaDataBatch
        meta_data_batch = MetaDataBatch()
        meta_data_batch.from_data_list(processed_data_list)

        # Return batch data
        return batch_task_data, meta_data_batch

    def data_process(self, task_data: TaskBase) -> MetaData:
        raise NotImplementedError("Subclass must implement this method")
