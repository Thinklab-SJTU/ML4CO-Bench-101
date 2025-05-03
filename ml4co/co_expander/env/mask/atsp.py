import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import atsp_greedy_decoder, to_numpy, to_tensor


def atsp_random_mask_dense(
    mask: Tensor, delta: float, raw_data: list
) -> Tensor:
    nodes_num = mask.shape[0]
    length = int(nodes_num * delta)
    ref_tour = raw_data[1]
    index = ref_tour[:length]
    mask[index, :] = True
    return mask


def atsp_update_mask_dense(
    heatmap: Tensor, e: Tensor, mask: Tensor, max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    # prepartion
    device = heatmap.device
    nodes_num = heatmap.shape[0]
    heatmap[range(nodes_num), range(nodes_num)] = 0
    heatmap: np.ndarray = to_numpy(heatmap)
    if continue_flag:
        e = heatmap
        mask[:] = True
    else:
        e[:] = 0
        mask[:] = False
        index = atsp_greedy_decoder(-heatmap)[:max_steps]
        e[index[:-1], index[1:]] = 1
        mask[index[:-1]] = True
        mask[torch.arange(e.shape[-1]), torch.arange(e.shape[-1])] = True
    
    return to_tensor(e).to(device), to_tensor(mask).to(device)