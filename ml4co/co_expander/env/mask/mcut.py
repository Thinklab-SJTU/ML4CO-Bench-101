import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import to_numpy


def mcut_random_mask_sparse(mask: Tensor, delta: float) -> Tensor:
    random_mask = torch.rand_like(mask.float()) < delta 
    return random_mask


def mcut_update_mask_sparse(
    heatmap: Tensor, x: Tensor, graph: Tensor, 
    mask: Tensor, max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    x = (heatmap > 0.5).int()
    if continue_flag:
        mask[:] = True
    else:
        heatmap = to_numpy(heatmap)
        sorted_nodes = np.argsort(-np.abs(heatmap-0.5))
        max_dn = mask.sum() + max_steps
        mask[sorted_nodes[:max_dn]] = True
    return x, mask