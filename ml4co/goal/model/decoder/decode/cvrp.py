import copy
import torch
import numpy as np
from torch import Tensor
from typing import Union, List, Any
from dataclasses import dataclass, asdict
from ml4co_kit import to_numpy, CVRPSolver, cvrp_classic_local_search
from goal.model.encoder.gat_encoder import GATEncoder
from goal.model.decoder.decode.utils import (
    remove_origin_and_reorder_tensor, remove_origin_and_reorder_matrix
)


@dataclass
class CVRPSubProblem:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    """
    e: Tensor
    x: Tensor
    demand: Tensor
    remain_cp: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    

def cvrp_greedy_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    return_cost: bool, knns: int, use_ls: bool
) -> Union[List[np.ndarray], np.floating]:
    # data info
    batch_size, nodes_num, _, _ = e.shape
    device = e.device
    
    # demands and cp
    demands = x[:, :, 0]
    remain_cp = torch.ones(size=(batch_size,)).to(device)
    
    # original index and init
    original_idxs = torch.arange(nodes_num)[None, :].repeat(batch_size, 1).to(device)
    tours = torch.zeros((batch_size, nodes_num)).long().to(device)
    via_depot_flags = torch.zeros((batch_size, nodes_num)).long().to(device)
    sub_problem = CVRPSubProblem(e, x, demands, remain_cp, original_idxs)
    
    # decoding
    if knns > 0:
        raise NotImplementedError()
    for dec_pos in range(1, nodes_num - 1):
        # get scores
        sub_problem.x[:, 0, 1] = sub_problem.remain_cp
        scores = model.forward(x=sub_problem.x, e=sub_problem.e, mask=None)
        scores = scores.reshape(batch_size, -1, 2)
        remain_cp = sub_problem.remain_cp
        scores[..., 0][sub_problem.demand > remain_cp.unsqueeze(-1)] = -torch.inf
        scores = scores.reshape(batch_size, -1)

        # select next node
        ori_selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
        selected_nodes = torch.div(ori_selected_nodes, 2, rounding_mode='trunc')
        via_depot = (ori_selected_nodes % 2 == 1)
        idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes)
        
        # update subproblem 
        subpb_size = sub_problem.e.shape[1]
        is_selected = (
            torch.arange(subpb_size)[None, ...].repeat(batch_size, 1).to(device) ==
            selected_nodes.repeat(1, subpb_size)
        )
        
        # 1. remain_cp
        cur_remain_cp = sub_problem.remain_cp
        action_demand = sub_problem.demand[np.arange(batch_size), selected_nodes.reshape(-1)]
        next_remain_cp = cur_remain_cp - action_demand
        via_cp = torch.ones_like(action_demand) - action_demand
        next_remain_cp[via_depot.bool().reshape(-1)] = via_cp[via_depot.bool().reshape(-1)]
        
        # 2. x
        x_is_selected = is_selected.unsqueeze(-1).repeat(1, 1, 2)
        new_beginning = sub_problem.x[x_is_selected].reshape(-1, 2).unsqueeze(dim=1)
        new_remaining = sub_problem.x[~is_selected].reshape((batch_size, -1, 2))[:, 1:]
        next_x = torch.cat([new_beginning, new_remaining], dim=1)
        
        # 3. e / demand / original index
        next_e = remove_origin_and_reorder_matrix(sub_problem.e, is_selected)
        next_demand = remove_origin_and_reorder_tensor(sub_problem.demand, is_selected)
        next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)

        # update subproblem 
        sub_problem = CVRPSubProblem(next_e, next_x, next_demand, next_remain_cp, next_original_idxs)

        # update solution
        tours[:, dec_pos] = idx_selected_original.squeeze(1) 
        via_depot_flags[:, dec_pos] = via_depot.squeeze(1) 

    # restruct tours
    restruct_tours = list()
    for tour, via_flag in zip(to_numpy(tours), to_numpy(via_depot_flags)):
        restruct_tour = list()
        for idx in range(nodes_num):
            if via_flag[idx]:
                restruct_tour.append(0)
            restruct_tour.append(tour[idx])
        restruct_tours.append(restruct_tour)
    
    # use_ls
    if use_ls:
        ls_tours = list()
        for greedy_tour, _raw_data in zip(restruct_tours, raw_data):
            ls_tour = cvrp_classic_local_search(
                init_tour=np.array(greedy_tour), depot=_raw_data[0][0],
                points=_raw_data[0][1:-1], demands=_raw_data[1][1:-1]
            )
            ls_tours.append(ls_tour)
        restruct_tours = ls_tours
          
    # check if return cost
    if return_cost:
        # raw data
        tmp_solver = CVRPSolver()
        depots_list = list()
        points_list = list()
        demands_list = list()
        capacities_list = list()
        for _raw_data in raw_data:
            depots_list.append(_raw_data[0][0])
            points_list.append(_raw_data[0][1:-1])
            demands_list.append(_raw_data[1][1:-1])
            capacities_list.append(1.0)
            
        # evaluate
        tmp_solver.from_data(
            depots=depots_list, points=points_list, demands=demands_list, 
            capacities=capacities_list, tours=restruct_tours
        )
        return tmp_solver.evaluate()
    else:
        return restruct_tours
    
    
def cvrp_beam_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    beam_size: int, return_cost: bool, knns: int, use_ls: bool
) -> Union[List[np.ndarray], np.floating]:
    # data info
    batch_size, nodes_num, _, _ = e.shape
    device = e.device
    
    # copy
    orig_distances = copy.deepcopy(e)
    
    # demands and cp
    demands = x[:, :, 0]
    remain_cp = torch.ones(size=(batch_size,)).to(device)
    
    # original index and init
    original_idxs = torch.arange(nodes_num)[None, :].repeat(batch_size, 1).to(device)
    tours = torch.zeros((batch_size * beam_size, nodes_num)).long().to(device)
    via_depots = torch.zeros((batch_size * beam_size, nodes_num)).bool().to(device)
    probabilities = torch.zeros((batch_size, 1), device=device)
    tour_lens = torch.zeros(size=(batch_size * beam_size, 1), device=device)
    sub_problem = CVRPSubProblem(e, x, demands, remain_cp, original_idxs)
    
    # decoding
    if knns > 0:
        raise NotImplementedError()
    for dec_pos in range(1, nodes_num - 1):
        cur_nodes_num = sub_problem.e.shape[1] * 2
        
        # get scores
        sub_problem.x[:, 0, 1] = sub_problem.remain_cp
        scores = model.forward(x=sub_problem.x, e=sub_problem.e, mask=None)
        if dec_pos == 1:
            scores = scores.reshape(batch_size, -1, 2)
        else:
            scores = scores.reshape(batch_size * beam_size, -1, 2)
        remain_cp = sub_problem.remain_cp
        scores[..., 0][sub_problem.demand > remain_cp.unsqueeze(-1)] = -torch.inf
        if dec_pos == 1:
            scores = scores.reshape(batch_size, -1)
        else:
            scores = scores.reshape(batch_size*beam_size, -1)
            
        # probability
        candidates = torch.softmax(scores, dim=1)
        probabilities = (probabilities.repeat(1, cur_nodes_num) + torch.log(candidates)).reshape(batch_size, -1)
            
        # select next node (1)
        num_instances = sub_problem.e.shape[0] // batch_size
        k = min(beam_size, probabilities.shape[1] - 2)
        topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
        batch_in_prev_input = ((num_instances * torch.arange(batch_size, device=device)).unsqueeze(dim=1) + \
                            torch.div(topk_indexes, cur_nodes_num, rounding_mode="floor")).flatten()
        topk_values = topk_values.flatten()
        topk_indexes = topk_indexes.flatten()
        probabilities = topk_values.unsqueeze(dim=1)
        
        # update sub_problem (1)
        sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
        sub_problem.e = sub_problem.e[batch_in_prev_input]
        sub_problem.demand = sub_problem.demand[batch_in_prev_input]
        sub_problem.x = sub_problem.x[batch_in_prev_input]
        sub_problem.remain_cp = sub_problem.remain_cp[batch_in_prev_input]
    
        # select next node (2)
        ori_selected_nodes = torch.remainder(topk_indexes, cur_nodes_num).unsqueeze(dim=1)
        selected_nodes = torch.div(ori_selected_nodes, 2, rounding_mode='trunc')
        via_depot = (ori_selected_nodes % 2 == 1)
        idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes).squeeze(-1)
        
        # update subproblem (2)
        subpb_size = sub_problem.e.shape[1]
        is_selected = (
            torch.arange(subpb_size)[None, ...].repeat(batch_size*beam_size, 1).to(device) ==
            selected_nodes.repeat(1, subpb_size)
        )
        
        # 1. remain_cp
        cur_remain_cp = sub_problem.remain_cp
        action_demand = sub_problem.demand[np.arange(batch_size*beam_size), selected_nodes.reshape(-1)]
        next_remain_cp = cur_remain_cp - action_demand
        via_cp = torch.ones_like(action_demand) - action_demand
        next_remain_cp[via_depot.bool().reshape(-1)] = via_cp[via_depot.bool().reshape(-1)]
        
        # 2. x
        x_is_selected = is_selected.unsqueeze(-1).repeat(1, 1, 2)
        new_beginning = sub_problem.x[x_is_selected].reshape(-1, 2).unsqueeze(dim=1)
        new_remaining = sub_problem.x[~is_selected].reshape((batch_size*beam_size, -1, 2))[:, 1:]
        next_x = torch.cat([new_beginning, new_remaining], dim=1)
        
        # 3. e / demand / original index
        next_e = remove_origin_and_reorder_matrix(sub_problem.e, is_selected)
        next_demand = remove_origin_and_reorder_tensor(sub_problem.demand, is_selected)
        next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)

        # update subproblem (3)
        sub_problem = CVRPSubProblem(next_e, next_x, next_demand, next_remain_cp, next_original_idxs)

        # update solution
        tours = tours[batch_in_prev_input]
        via_depots = via_depots[batch_in_prev_input]
        tours[:, dec_pos] = idx_selected_original
        via_depot = via_depot.reshape(-1)
        via_depots[:, dec_pos] = via_depot
        tour_lens = tour_lens[batch_in_prev_input]
        
        # compute lenghts for direct edges
        direct_length = orig_distances[batch_in_prev_input, tours[:, dec_pos], tours[:, dec_pos - 1], 0]
        tour_lens[~via_depots[:, dec_pos]] += direct_length[~via_depots[:, dec_pos]].reshape(-1, 1)

        # compute lenghts for edges via depot
        via_length = orig_distances[batch_in_prev_input, tours[:, dec_pos-1], 0, 0] + \
            orig_distances[batch_in_prev_input, 0, tours[:, dec_pos], 0]
        tour_lens[via_depots[:, dec_pos]] += via_length[via_depots[:, dec_pos]].reshape(-1, 1)

        # update distances (beam)
        orig_distances = orig_distances[batch_in_prev_input]

    tour_lens += orig_distances[batch_in_prev_input, tours[:, dec_pos-1], 0, 0].reshape(-1, 1)
    distances = tour_lens.reshape(batch_size, -1)
    batch_idx = torch.argmin(distances, dim=1) + torch.arange(batch_size).to(device) * beam_size
    best_tours = tours[batch_idx]
    best_via_depots = via_depots[batch_idx]

    # restruct tours
    restruct_tours = list()
    for tour, via_flag in zip(to_numpy(best_tours), to_numpy(best_via_depots)):
        restruct_tour = list()
        for idx in range(nodes_num):
            if via_flag[idx]:
                restruct_tour.append(0)
            restruct_tour.append(tour[idx])
        restruct_tours.append(restruct_tour)  

    # use_ls
    if use_ls:
        ls_tours = list()
        for greedy_tour, _raw_data in zip(restruct_tours, raw_data):
            ls_tour = cvrp_classic_local_search(
                init_tour=np.array(greedy_tour), depot=_raw_data[0][0],
                points=_raw_data[0][1:-1], demands=_raw_data[1][1:-1]
            )
            ls_tours.append(ls_tour)
        restruct_tours = ls_tours
          
    # check if return cost
    if return_cost:
        # raw data
        tmp_solver = CVRPSolver()
        depots_list = list()
        points_list = list()
        demands_list = list()
        capacities_list = list()
        for _raw_data in raw_data:
            depots_list.append(_raw_data[0][0])
            points_list.append(_raw_data[0][1:-1])
            demands_list.append(_raw_data[1][1:-1])
            capacities_list.append(1.0)
            
        # evaluate
        tmp_solver.from_data(
            depots=depots_list, points=points_list, demands=demands_list, 
            capacities=capacities_list, tours=restruct_tours
        )
        return tmp_solver.evaluate()
    else:
        return restruct_tours