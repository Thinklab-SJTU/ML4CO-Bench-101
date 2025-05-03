import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import to_numpy, to_tensor


def mvc_random_mask_sparse(mask: Tensor, delta: float) -> Tensor:
    random_mask = torch.rand_like(mask.float()) < delta 
    return random_mask


def mvc_update_mask_sparse(
    heatmap: Tensor, x: Tensor, graph: Tensor, 
    mask: Tensor, max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    # perpartion
    device = heatmap.device
    heatmap = to_numpy(heatmap)
    graph = to_numpy(graph)
    mask = to_numpy(mask)
    x = to_numpy(x.bool())
    sorted_nodes = np.argsort(heatmap)
     
    # deal with different continue flag
    if continue_flag:
        for node in sorted_nodes:
            if not mask[node]:
                connect_nodes = np.where(graph[node] == 1)[0]
                x[connect_nodes] = True
                mask[connect_nodes] = True
                x[node] = False
                mask[node] = True
    else:
        max_dn = mask.sum() + max_steps
        for node in sorted_nodes:
            if mask.sum() > max_dn:
                break
            if not mask[node]:
                connect_nodes = np.where(graph[node] == 1)[0]
                x[connect_nodes] = True
                mask[connect_nodes] = True
                x[node] = False
                mask[node] = True

    return to_tensor(x).int().to(device), to_tensor(mask).to(device)