import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from sklearn.neighbors import KDTree
from ml4co_kit import check_dim, to_tensor


def cvrp_sparse_process(
    depot: np.ndarray, points: np.ndarray, demand: np.ndarray, 
    ref_tour: np.ndarray, sparse_factor: int, device: str
) -> Sequence[Tensor]:
    raise NotImplementedError()