import os
import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple
from torch.utils.data import Dataset
from ml4co_kit import WrapperBase, TaskBase


class MetaData(object):
    def __init__(
        self, 
        node_feature: Tensor, 
        edge_feature: Tensor, 
        edge_index: Tensor,
        ground_truth: Tensor
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
    Batched sparse graph data, implemented with pure torch
    (replacement for torch_geometric.data.Batch).
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
        """
        # Initialize lists
        node_features = []
        edge_features = []
        edge_indices = []
        ground_truths = []
        batch_vecs = []
        ptr = [0]
        node_offset = 0

        # Process data
        for i, data in enumerate(data_list):
            num_nodes = data.node_feature.size(0)
            node_features.append(data.node_feature)
            edge_features.append(data.edge_feature)
            edge_indices.append(data.edge_index + node_offset)
            ground_truths.append(data.ground_truth)
            batch_vecs.append(
                torch.full((num_nodes,), i, dtype=torch.long, 
                device=data.node_feature.device)
            )
            node_offset += num_nodes
            ptr.append(node_offset)

        # Update attributes
        self.node_feature = torch.cat(node_features, dim=0)
        self.edge_feature = torch.cat(edge_features, dim=0)
        self.edge_index = torch.cat(edge_indices, dim=1)
        self.ground_truth = torch.cat(ground_truths, dim=0)
        self.batch = torch.cat(batch_vecs, dim=0)
        self.ptr = torch.tensor(
            ptr, dtype=torch.long, device=data_list[0].node_feature.device
        )

    def to_cuda(self):
        self.node_feature = self._to_cuda(self.node_feature)
        self.edge_feature = self._to_cuda(self.edge_feature)
        self.edge_index = self._to_cuda(self.edge_index)
        self.ground_truth = self._to_cuda(self.ground_truth)
        self.batch = self._to_cuda(self.batch)
        self.ptr = self._to_cuda(self.ptr)

    @staticmethod
    def _to_cuda(tensor: Tensor) -> Tensor:
        return None if tensor is None else tensor.cuda()


class MetaDataset(Dataset):
    def __init__(
        self, 
        wrapper: WrapperBase, 
        num_per_epoch: int,
        data_path: str = None,
        data_folder: str = None,
    ):
        # Super Args
        super(MetaDataset, self).__init__()

        # Set Attributes (Data related)
        self.wrapper = wrapper
        self.num_per_epoch = num_per_epoch
        self.data_folder = data_folder
        
        # Data list
        self.data_list = list()

        # Process data path
        if data_folder is not None:
            self.sub_files = [
                os.path.join(self.data_folder, file) \
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
        task_data = self.data_list[self.left_num-1]
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
    