import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import to_numpy, to_tensor


def tsp_random_mask_dense(mask: Tensor, delta: float) -> Tensor:
    random_mask = torch.rand_like(mask.float()) < delta 
    return random_mask


def tsp_random_mask_sparse(mask: Tensor, delta: float) -> Tensor:
    random_mask = torch.rand_like(mask.float()) < delta 
    return random_mask


def tsp_update_mask_dense(
    heatmap: Tensor, e: Tensor, mask: Tensor, 
    max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    # prepartion
    device = heatmap.device
    heatmap: np.ndarray = to_numpy(heatmap)

    # deal with different continue flag
    if continue_flag:
        e = heatmap
        mask[:] = True
    else:
        heatmap = (heatmap + heatmap.T) / 2
        index = np.argsort(-heatmap.reshape(-1))[:max_steps]
        mask = mask.reshape(-1)
        mask[:] = False
        mask[index] = True
        mask = mask.reshape(e.shape[0], e.shape[1])
        e[:] = 1
        if mask.all():
           e = heatmap
                
    return to_tensor(e).to(device), to_tensor(mask).to(device)
    

def tsp_update_mask_sparse(
    heatmap: Tensor, e: Tensor, mask: Tensor, 
    max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    # device
    device = heatmap.device
    
    # deal with different continue flag
    if continue_flag:
        e = heatmap
        mask[:] = True
    else:
        heatmap = to_numpy(heatmap)
        index = np.argsort(-heatmap)[:max_steps]
        mask[:] = False
        mask[index] = True
        e[:] = 1

    return to_tensor(e).to(device), to_tensor(mask).to(device)