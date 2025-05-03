import torch
import numpy as np
from typing import List, Union, Any
from ml4co_kit import (
    check_dim, MClGraphData, MCutGraphData, 
    MISGraphData, MVCGraphData
)
from gnn4co.env.dense import (
    atsp_dense_process, cvrp_dense_process, tsp_dense_process
)


class GNN4CODenser(object):
    def __init__(self, device: str) -> None:
        self.device = device
    
    #################################
    #        Raw Data Process       #
    #################################
    
    def initial_lists(self):
        self.nodes_feature_list = list()
        self.x_list = list()
        self.graph_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        
    def update_lists(self, dense_data: Any):
        self.x_list.append(dense_data[0])
        self.graph_list.append(dense_data[1])
        self.ground_truth_list.append(dense_data[2])
        self.nodes_num_list.append(dense_data[3])
    
    def edge_merge_process(self, task: str, with_gt: bool) -> Any:
        # nodes feature
        if self.x_list[0] is not None:
            x = torch.stack(self.x_list, 0).to(self.device)
        else:
            x = None
            
        # graph
        graph = torch.stack(self.graph_list, 0).to(self.device)

        # ground truth
        if with_gt:
            ground_truth = torch.stack(
                self.ground_truth_list, 0
            ).to(self.device) # (B, V, V) or (B, V)
        else:
            ground_truth = None
        
        return (task, x, graph, ground_truth, self.nodes_num_list)  
        
    def node_merge_process(self, task: str, with_gt: bool) -> Any:
        raise NotImplementedError()
    
    def atsp_batch_data_process(
        self, dists: np.ndarray, ref_tours: np.ndarray, sampling_num: int = 1
    ) -> Any:
        # check dimension
        check_dim(dists, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # dense process
        for idx in range(dists.shape[0]):
            dense_data = atsp_dense_process(
                dists=dists[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
            
        # merge
        return self.edge_merge_process(
            task="ATSP", 
            with_gt=True if ref_tours is not None else False
        )
           
    def cvrp_batch_data_process(
        self, 
        depots: Union[np.ndarray, list, List[np.ndarray]], 
        points: Union[np.ndarray, list, List[np.ndarray]], 
        demands: Union[np.ndarray, list, List[np.ndarray]], 
        ref_tours: Union[np.ndarray, list, List[np.ndarray]], 
        sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # dense process
        for idx in range(points.shape[0]):
            dense_data = cvrp_dense_process(
                depot=depots[idx], 
                points=points[idx], 
                demand=demands[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.edge_merge_process(
            task="CVRP", 
            with_gt=True if ref_tours is not None else False
        )

    def mcl_batch_data_process(
        self, graph_data: List[MClGraphData], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError()
    
    def mcut_batch_data_process(
        self, graph_data: List[MCutGraphData], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError()

    def mis_batch_data_process(
        self, graph_data: List[MISGraphData], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError()

    def mvc_batch_data_process(
        self, graph_data: List[MVCGraphData], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError()
    
    def tsp_batch_data_process(
        self, points: np.ndarray, ref_tours: np.ndarray, sampling_num: int = 1
    ) -> Any:
        # check dimension
        check_dim(points, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # dense process
        for idx in range(points.shape[0]):
            dense_data = tsp_dense_process(
                points=points[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.edge_merge_process(
            task="TSP", 
            with_gt=True if ref_tours is not None else False
        )