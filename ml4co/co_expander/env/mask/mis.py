import copy
import torch
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit import to_numpy, to_tensor


def mis_random_mask_sparse(
    graph: Tensor, mask: Tensor, ground_truth: Tensor, delta: float
) -> Tensor:
    random_mask = torch.rand_like(mask.float()) < delta 
    true_index_list = torch.where(random_mask == True)[0]
    for true_index in true_index_list:
        if ground_truth[true_index] == 1:
            conn_nodes = torch.where(graph[true_index] == 1)[0]
            random_mask[conn_nodes] = True
            random_mask[true_index] = True
    return random_mask


def mis_update_mask_sparse(
    heatmap: Tensor, x: Tensor, graph: Tensor, mask: Tensor, 
    max_steps: int, continue_flag: bool, beam_size: int = -1
) -> Sequence[Tensor]:
    # perpartion
    device = heatmap.device
    heatmap = to_numpy(heatmap)
    graph = to_numpy(graph)
    mask = to_numpy(mask)
    x: np.ndarray = to_numpy(x.bool())
    sorted_nodes = np.argsort(-heatmap)
    
    # deal with different continue flag
    if continue_flag:
        if beam_size > 0:
            if x.sum() == 0:
                empty_flag = [True for _ in range(beam_size)]
                independent_set = [list() for _ in range(beam_size)]     
            else:
                empty_flag = [False for _ in range(beam_size)]
                independent_set = [np.where(x==1).tolist()]
            sol = np.repeat(x.astype(np.int32).reshape(1, -1), beam_size, axis=0)
            
            sorted_nodes = np.argsort(-heatmap)
            for node in sorted_nodes:
                if mask[node]:
                    continue
                for idx in range(beam_size):
                    if empty_flag[idx]:
                        independent_set[idx].append(node)
                        sol[idx][node] = 1
                        empty_flag[idx] = False
                        break
                    if (graph[node][independent_set[idx]] == 0).all():
                        independent_set[idx].append(node)
                        sol[idx][node] = 1
                        break
            best_idx = np.argmax(sol.sum(axis=1))
            x = sol[best_idx]
            mask[:] = True
        else:                  
            for node in sorted_nodes:
                if not mask[node]:
                    if (graph[node][x]).sum() == 0:
                        connect_nodes = np.where(graph[node] == 1)[0]
                        x[connect_nodes] = False
                        x[node] = True
                        mask[connect_nodes] = True
                        mask[node] = True
                        
    else:
        max_dn = mask.sum() + max_steps
        ori_mask = copy.deepcopy(mask)
        for node in sorted_nodes:
            if mask.sum() > max_dn:
                break
            if not ori_mask[node]:
                if (graph[node][x]).sum() == 0:
                    connect_nodes = np.where(graph[node] == 1)[0]
                    x[connect_nodes] = False
                    x[node] = True
                    mask[connect_nodes] = True
                    mask[node] = True
                else:
                    break
                
    return to_tensor(x).int().to(device), to_tensor(mask).to(device)