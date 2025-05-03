import torch
import numpy as np
from torch import Tensor
from ml4co_kit import to_numpy
from typing import Union, List, Any
from dataclasses import dataclass, asdict
from goal.model.encoder.gat_encoder import GATEncoder
    

@dataclass
class MVCSubProblem:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    e: Tensor
    original_idxs: Tensor
    mask: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    

def mvc_greedy_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    return_cost: bool, knns: int
) -> Union[List[np.ndarray], np.floating]:
    # data info
    batch_size, nodes_num, _ = e.shape
    device = e.device
    
    # original index and init
    original_idxs = torch.arange(nodes_num)[None, :].repeat(batch_size, 1).to(device)
    sols = -torch.ones((batch_size, nodes_num)).long().to(device)
    mask = torch.zeros_like(e).bool()
    mask[e.sum(dim=-1) == 0] = True
    nodes_num = mask.shape[-1]
    mask[:, torch.arange(nodes_num), torch.arange(nodes_num)] = True
    sub_pb = MVCSubProblem(e=e, original_idxs=original_idxs, mask=mask)
    
    # decoding
    for dec_pos in range(nodes_num):
        
        # inference
        scores = model.forward(x=None, e=sub_pb.e, mask=sub_pb.mask)

        # select next node
        selected_nodes = torch.argmin(scores, dim=1, keepdim=True)
        idx_selected_original = torch.gather(sub_pb.original_idxs, 1, selected_nodes)
        idx_selected_original[scores.min(dim=-1)[0] == torch.inf] = -1
        
        # update subproblem
        subpb_size = sub_pb.e.shape[1]
        is_selected = (
            torch.arange(subpb_size)[None, ...].repeat(batch_size, 1).to(device) ==
            selected_nodes.repeat(1, subpb_size)
        )

        next_e = sub_pb.e[~is_selected].reshape((batch_size, -1, subpb_size))
        next_e = next_e.transpose(1, 2)[~is_selected].reshape((batch_size, -1, subpb_size-1))
        next_e = next_e.transpose(1, 2)

        mask = sub_pb.mask
        mask[sub_pb.e[torch.arange(batch_size), selected_nodes.squeeze(-1)] == 1.] = True
        next_mask = mask[~is_selected].reshape((batch_size, -1, subpb_size))
        next_mask = next_mask.transpose(1, 2)[~is_selected].reshape((batch_size, -1, subpb_size-1))
        next_mask = next_mask.transpose(1, 2)

        next_original_idxs = sub_pb.original_idxs[~is_selected].reshape((batch_size, -1))
        sub_pb = MVCSubProblem(
            e=next_e, original_idxs=next_original_idxs, mask=next_mask
        ) 

        # update solution
        sols[:, dec_pos] = idx_selected_original.squeeze(1)
        
        # break
        if mask.all():
            break

    # np.ndarray
    sols = to_numpy(sols)

    # check if return cost
    if return_cost:
        nodes_num_list = [raw_data[idx].nodes_num for idx in range(len(raw_data))]
        nodes_num_array = np.array(nodes_num_list)
        unsel_nodes_num = np.sum(sols != -1)
        sel_nodes_num = np.sum(nodes_num_array) - unsel_nodes_num
        return sel_nodes_num / batch_size
    else:
        return sols


def mvc_beam_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    beam_size: int, return_cost: bool
) -> Union[List[np.ndarray], np.floating]:
    raise NotImplementedError()