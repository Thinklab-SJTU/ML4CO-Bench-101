import torch
import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import (
    to_numpy, to_tensor, tsp_mcts_local_search, TSPSolver, 
    np_sparse_to_dense, tsp_greedy_decoder, tsp_2opt_local_search
)


def tsp_dense_decode(
    vars: Tensor, nodes_feature: Tensor, return_cost: bool, use_2opt: bool = True
) -> Union[List[np.ndarray], np.floating]:
    vars = to_numpy(vars)
    vars = (vars + vars.transpose(0, 2, 1)) / 2
    nodes_feature = to_numpy(nodes_feature)
    
    # from vars to decoding the tours
    tours = list()
    for idx in range(vars.shape[0]):
        heatmap = vars[idx]
        greedy_tour = tsp_greedy_decoder(heatmap)
        tours.append(greedy_tour)
    tours = np.array(tours)

    # 2opt
    if use_2opt:
        bs = tours.shape[0]
        ls_tours = tsp_mcts_local_search(
            init_tours=tours,
            heatmap=vars, 
            points=nodes_feature,
            time_limit=0.0, # set time limit as 0.0 is equal to 2opt
            type_2opt=2
        )
        if bs == 1:
            ls_tours = np.expand_dims(ls_tours, 0)
        tours = ls_tours
    
    if return_cost:
        tmp_solver = TSPSolver()
        tmp_solver.from_data(points=nodes_feature, tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours
    
    
def tsp_sparse_decode(
    vars: Tensor, nodes_feature: Tensor, edge_index: Tensor, nodes_num_list: list, 
    edges_num_list: list, return_cost: bool, use_2opt: bool = True
) -> Union[List[np.ndarray], np.floating]:
    # preparation
    device = vars.device
    heatmap = to_numpy(vars)
    edge_index = to_numpy(edge_index)
    nodes_feature = to_numpy(nodes_feature)
    
    # from heatmap to decoding the tours
    tours = list()
    x_begin_idx = 0
    e_begin_idx = 0
    for nodes_num, edges_num in zip(nodes_num_list, edges_num_list):
        # preparation
        x_end_idx = x_begin_idx + nodes_num
        e_end_idx = e_begin_idx + edges_num
        _heatmap = heatmap[e_begin_idx:e_end_idx]
        _points = nodes_feature[x_begin_idx:x_end_idx]
        _edge_index = edge_index[:, e_begin_idx:e_end_idx] - x_begin_idx
        
        # heatmap sparse to dense
        _heatmap_dense = np_sparse_to_dense(
            nodes_num=nodes_num, edge_index=_edge_index, edge_attr=_heatmap
        )
        _heatmap = (_heatmap_dense + _heatmap_dense.T) / 2
        
        # greedy decoding
        greedy_tour = tsp_greedy_decoder(heatmap=_heatmap)
        
        # 2OPT
        if use_2opt:
            greedy_tour = tsp_2opt_local_search(
                init_tours=greedy_tour, points=_points, device=device
            )
        
        # update index
        x_begin_idx = x_end_idx
        e_begin_idx = e_end_idx
        tours.append(greedy_tour)

    # format
    tours = np.array(tours)

    if return_cost:
        tmp_solver = TSPSolver()
        tmp_solver.from_data(points=nodes_feature, tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours