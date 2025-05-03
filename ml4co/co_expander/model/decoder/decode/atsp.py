import torch
import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import to_numpy, np_sparse_to_dense, tsp_greedy_decoder
from ml4co_kit import ATSPSolver, atsp_2opt_local_search, atsp_greedy_decoder


def atsp_dense_decode(
    vars: Tensor, graph: Tensor, return_cost: bool, use_2opt: bool = True
) -> Union[List[np.ndarray], np.floating]:
    vars = to_numpy(vars)
    graph = to_numpy(graph)

    # from vars to decoding the tours
    tours = list()
    for idx in range(vars.shape[0]):
        heatmap = vars[idx]
        greedy_tour = atsp_greedy_decoder(-heatmap)
        tours.append(greedy_tour)
    tours = np.array(tours)

    # 2opt
    if use_2opt:
        bs = tours.shape[0]
        ls_tours = atsp_2opt_local_search(
            init_tours=tours, dists=graph
        )
        if bs == 1:
            ls_tours = np.expand_dims(ls_tours, 0)
        tours = ls_tours
    
    if return_cost:
        tmp_solver = ATSPSolver()
        tmp_solver.from_data(dists=graph, tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours
    
    
def atsp_sparse_decode(
    vars: Tensor, nodes_feature: Tensor, edges_feature: Tensor, 
    edge_index: Tensor, graph_list: List[Tensor], nodes_num_list: list, 
    edges_num_list: list, return_cost: bool
) -> Union[List[np.ndarray], np.floating]:
    heatmap = to_numpy(vars)
    edge_index = to_numpy(edge_index)
    nodes_feature = to_numpy(nodes_feature)

    # from heatmap to decoding the tours
    tours = list()
    x_begin_idx = 0
    e_begin_idx = 0
    for nodes_num, edges_num, graph in zip(nodes_num_list, edges_num_list, graph_list):
        # preparation
        x_end_idx = x_begin_idx + nodes_num
        e_end_idx = e_begin_idx + edges_num
        _heatmap = heatmap[e_begin_idx:e_end_idx]
        _edge_index = edge_index[:, e_begin_idx:e_end_idx] - x_begin_idx
        
        # heatmap sparse to dense
        _heatmap = np_sparse_to_dense(
            nodes_num=nodes_num, edge_index=_edge_index, edge_attr=_heatmap
        )
        
        # greedy decoding
        greedy_tour = atsp_greedy_decoder(heatmap=-_heatmap)
        
        # 2OPT
        ls_tour = atsp_2opt_local_search(init_tours=greedy_tour, dists=to_numpy(graph))
        
        # update index
        x_begin_idx = x_end_idx
        e_begin_idx = e_end_idx
        tours.append(ls_tour)

    # format
    tours = np.array(tours)

    if return_cost:
        tmp_solver = ATSPSolver()
        tmp_solver.from_data(dists=to_numpy(graph_list), tours=tours)
        return tmp_solver.evaluate()
    else:
        return tours
    