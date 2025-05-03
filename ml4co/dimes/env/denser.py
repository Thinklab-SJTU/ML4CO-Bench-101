import torch
import numpy as np
from typing import List, Union, Any
from ml4co_kit import check_dim
from dimes.env.dense import tsp_dense_process


class DIMESDenser(object):
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