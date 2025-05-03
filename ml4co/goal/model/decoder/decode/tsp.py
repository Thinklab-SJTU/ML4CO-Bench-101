import copy
import torch
import numpy as np
from torch import Tensor
from typing import Union, List, Any
from dataclasses import dataclass, asdict
from ml4co_kit import to_numpy, TSPSolver, tsp_2opt_local_search
from goal.model.encoder.gat_encoder import GATEncoder
from goal.model.decoder.decode.utils import (
    remove_origin_and_reorder_tensor, remove_origin_and_reorder_matrix
)


@dataclass
class TSPSubProblem:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    """
    e: Tensor
    original_idxs: Tensor

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
    

def tsp_greedy_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    return_cost: bool, knns: int, use_2opt: bool
) -> Union[List[np.ndarray], np.floating]:
    # data info
    batch_size, nodes_num, _, _ = e.shape
    device = e.device
    
    # original index and init
    original_idxs = torch.arange(nodes_num)[None, :].repeat(batch_size, 1).to(device)
    tours = torch.zeros((batch_size, nodes_num)).long().to(device)
    sub_problem = TSPSubProblem(e, original_idxs)

    # decoding
    for dec_pos in range(1, nodes_num - 1):
        cur_nodes_num = sub_problem.e.shape[1]
        
        # inference to get score
        if 0 < knns < cur_nodes_num:
            # sort node by distance from the origin (ignore the target node)
            dist_matrices = sub_problem.e
            _, sorted_nodes_idx = torch.sort(dist_matrices[:, 0, :-1, 0], dim=-1)

            # select KNNs and add the target at the end
            knn_indices = sorted_nodes_idx[:, :knns-1] 
            knn_indices = torch.cat([
                knn_indices,
                torch.full([batch_size, 1], cur_nodes_num - 1).to(device)
            ], dim=-1)

            # knn dist
            knn_dist_matrices = torch.gather(
                dist_matrices, 1, knn_indices[..., None, None].repeat(1, 1, cur_nodes_num, 2)
            )
            knn_dist_matrices = torch.gather(
                knn_dist_matrices, 2, knn_indices[:, None, :, None].repeat(1, knns, 1, 2)
            )
            right_term = knn_dist_matrices.amax(dim=-1).amax(dim=-1).amax(dim=-1)
            knn_dist_matrices = (
                knn_dist_matrices / right_term[:, None, None, None].repeat(1, knns, knns, 2)
            )
            knn_scores = model.forward(x=None, e=knn_dist_matrices)
            scores = torch.full((batch_size, cur_nodes_num), -np.inf).to(device)
            scores = torch.scatter(scores, 1, knn_indices, knn_scores)
            
        else:
            scores = model.forward(x=None, e=sub_problem.e)

        # select next node
        selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
        idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes)

        # update subproblem
        subpb_size = sub_problem.e.shape[1]
        is_selected = (
            torch.arange(subpb_size)[None, ...].repeat(batch_size, 1).to(device) ==
            selected_nodes.repeat(1, subpb_size)
        )
        next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)
        next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.e, is_selected)

        sub_problem = TSPSubProblem(e=next_dist_matrices, original_idxs=next_original_idxs)
        
        # update solution
        tours[:, dec_pos] = idx_selected_original.squeeze(1)   
    
    # 2opt
    tours = to_numpy(tours)
    if use_2opt:
        points = to_numpy(raw_data)[:, :-1, :]
        ls_tours = list()
        for _points, _tour in zip(points, tours):
            ls_tours.append(tsp_2opt_local_search(
                init_tours=_tour, points=_points, device=device
            ))
        tours = to_numpy(ls_tours)
        
    # check if return cost
    if return_cost:
        tmp_solver = TSPSolver()
        tmp_solver.from_data(points=raw_data, tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours


def tsp_beam_decode(
    x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
    beam_size: int, return_cost: bool, knns: int, use_2opt: bool
) -> Union[List[np.ndarray], np.floating]:
    # data info
    batch_size, nodes_num, _, _ = e.shape
    device = e.device
    
    # copy
    orig_distances = copy.deepcopy(e)
    
    # original index and init
    original_idxs = torch.arange(nodes_num)[None, :].repeat(batch_size, 1).to(device)
    tours = torch.zeros((batch_size*beam_size, nodes_num)).long().to(device)
    probabilities = torch.zeros((batch_size, 1), device=device)
    tour_lens = torch.zeros(size=(batch_size * beam_size, 1), device=device)
    sub_problem = TSPSubProblem(e, original_idxs)
    
    # decoding
    for dec_pos in range(1, nodes_num - 1):
        cur_nodes_num = sub_problem.e.shape[1]
        
        # inference to get score
        if 0 < knns < cur_nodes_num:
            # sort node by distance from the origin (ignore the target node)
            dist_matrices = sub_problem.e
            _, sorted_nodes_idx = torch.sort(dist_matrices[:, 0, :-1, 0], dim=-1)

            # select KNNs and add the target at the end
            knn_indices = sorted_nodes_idx[:, :knns-1]
            knn_indices = torch.cat([
                knn_indices,
                torch.full([knn_indices.shape[0], 1], cur_nodes_num - 1).to(device)
            ], dim=-1)

            # knn dist
            knn_dist_matrices = torch.gather(
                dist_matrices, 1, knn_indices[..., None, None].repeat(1, 1, cur_nodes_num, 2)
            )
            knn_dist_matrices = torch.gather(
                knn_dist_matrices, 2, knn_indices[:, None, :, None].repeat(1, knns, 1, 2)
            )
            right_term = knn_dist_matrices.amax(dim=-1).amax(dim=-1).amax(dim=-1)
            knn_dist_matrices = (
                knn_dist_matrices / right_term[:, None, None, None].repeat(1, knns, knns, 2)
            )
            knn_scores = model.forward(x=None, e=knn_dist_matrices)
            scores = torch.full((knn_scores.shape[0], cur_nodes_num), -np.inf).to(device)
            scores = torch.scatter(scores, 1, knn_indices, knn_scores)
            
        else:
            scores = model.forward(x=None, e=sub_problem.e)

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
        selected_nodes = torch.remainder(topk_indexes, cur_nodes_num).unsqueeze(dim=1)
        idx_selected_original = torch.gather(sub_problem.original_idxs, 1, selected_nodes).squeeze(-1)
        
        # update subproblem (2)
        subpb_size = sub_problem.e.shape[1]
        is_selected = (
            torch.arange(subpb_size)[None, ...].repeat(beam_size*batch_size, 1).to(device) ==
            selected_nodes.repeat(1, subpb_size)
        )
        next_original_idxs = remove_origin_and_reorder_tensor(sub_problem.original_idxs, is_selected)
        next_dist_matrices = remove_origin_and_reorder_matrix(sub_problem.e, is_selected)

        sub_problem = TSPSubProblem(e=next_dist_matrices, original_idxs=next_original_idxs)

        # update solution
        tours = tours[batch_in_prev_input]
        tours[:, dec_pos] = idx_selected_original
        tour_lens = tour_lens[batch_in_prev_input]
        tour_lens += orig_distances[
            batch_in_prev_input, tours[:, dec_pos-1], tours[:, dec_pos], 0
        ].unsqueeze(-1)
        orig_distances = orig_distances[batch_in_prev_input]   
        
    tour_lens += orig_distances[batch_in_prev_input, tours[:, dec_pos-1], 0, 0].unsqueeze(-1)
    distances = tour_lens.reshape(batch_size, -1)
    tours = tours.reshape(batch_size, -1, nodes_num)
    tours = tours[torch.arange(batch_size), torch.argmin(distances, dim=1)]
    
    # 2opt
    tours = to_numpy(tours)
    if use_2opt:
        points = to_numpy(raw_data)[:, :-1, :]
        ls_tours = list()
        for _points, _tour in zip(points, tours):
            ls_tours.append(
                tsp_2opt_local_search(
                    init_tours=_tour, 
                    points=_points,
                    device=device
                )
            )
        tours = to_numpy(ls_tours)
        
    # check if return cost
    if return_cost:
        tmp_solver = TSPSolver()
        tmp_solver.from_data(points=raw_data, tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours