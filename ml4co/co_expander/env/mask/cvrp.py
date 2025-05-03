import copy
import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import (
    cvrp_classic_local_search, to_numpy, to_tensor, cvrp_greedy_decoder
)


def cvrp_random_mask_dense(
    mask: Tensor, delta: float, raw_data: list
) -> Tensor:
    random_mask = copy.deepcopy(mask)
    ref_tour = raw_data[2]
    split_tours = np.split(ref_tour, np.where(ref_tour == 0)[0])[1: -1]
    all_size = len(split_tours)
    hint_size = int(all_size * delta)
    if hint_size > 0:
        for idx in range(hint_size):
            tour = split_tours[idx]
            mask[tour[1:]] = True
            mask[:, tour[1:]] = True
            mask[0][tour[1]] = True
    random_mask = torch.rand_like(mask.float()) < delta 
    return random_mask


def cvrp_update_mask_dense(
    heatmap: Tensor, e: Tensor, mask: Tensor, raw_data: list, 
    max_steps: int, continue_flag: bool
) -> Sequence[Tensor]:
    # continue
    if continue_flag:
        mask[:] = True
        return heatmap, mask
    
    # prepartion
    mask[:] = False
    e[:] = 0
    device = heatmap.device
    heatmap = to_numpy(heatmap)

    # greedy decoding and local search
    coords, demand, _ = raw_data
    demand: np.ndarray
    demand = demand.reshape(-1)
    greedy_tour = cvrp_greedy_decoder(
        heatmap=copy.deepcopy(heatmap), norm_demand=demand
    )
    ls_tour = cvrp_classic_local_search(
        init_tour=greedy_tour, depot=coords[0], points=coords[1:], demands=demand
    )
    
    # update mask and determination variables
    split_tours = np.split(ls_tour, np.where(ls_tour == 0)[0])[1: -1]
    for idx in range(max_steps):
        fix_tour = split_tours[idx]
        e[fix_tour[:-1], [fix_tour[1:]]] = 1
        e[[fix_tour[1:]], fix_tour[:-1]] = 1
        mask[fix_tour[1:], :] = True
        mask[:, fix_tour[1:]] = True
        mask[0][fix_tour[1]] = True
        mask[fix_tour[1]][0] = True
    
    return to_tensor(e).to(device), to_tensor(mask).to(device)