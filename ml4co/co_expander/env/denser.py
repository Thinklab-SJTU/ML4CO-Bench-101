import torch
import numpy as np
from torch import Tensor
from ml4co_kit import check_dim
from typing import List, Union, Any
from co_expander.env.processor import DataProcessor
from co_expander.env.dense import (
    atsp_dense_process, cvrp_dense_process, tsp_dense_process
)
from co_expander.env.mask import (
    atsp_random_mask_dense, atsp_update_mask_dense,
    cvrp_random_mask_dense, cvrp_update_mask_dense,
    tsp_random_mask_dense, tsp_update_mask_dense,
)


class COExpanderDenser(DataProcessor):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
    
    #################################
    #        Raw Data Process       #
    #################################
    
    def initial_lists(self):
        self.nodes_feature_list = list()
        self.x_list = list()
        self.graph_list = list()
        self.e_list = list()
        self.mask_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        self.raw_data_list = list()
        
    def update_lists(self, dense_data: Any):
        self.nodes_feature_list.append(dense_data[0])
        self.x_list.append(dense_data[1])
        self.graph_list.append(dense_data[2])
        self.e_list.append(dense_data[3])
        self.mask_list.append(dense_data[4])
        self.ground_truth_list.append(dense_data[5])
        self.nodes_num_list.append(dense_data[6])
        self.raw_data_list.append(dense_data[7])
    
    def merge_process(self, task: str, with_gt: bool) -> Any:
        # [0] nodes' feature
        if self.nodes_feature_list[0] is not None:
            nodes_feature = torch.stack(self.nodes_feature_list, 0).to(self.device)
        else:
            nodes_feature = None
            
        # [1] nodes' decision variable
        x = None
        
        # [2] edges' feature
        graph = torch.stack(self.graph_list, 0).to(self.device)
        
        # [3] edges' decision variable
        e = torch.stack(self.e_list, 0).to(self.device)
        
        # [4] mask
        mask = torch.stack(self.mask_list, 0).to(self.device)
        
        # [5] ground truth
        if with_gt:
            ground_truth = torch.stack(self.ground_truth_list, 0).to(self.device) # (B, V, V) or (B, V)
        else:
            ground_truth = None
        
        # return processed dense data (9 items)
        return (
            task, nodes_feature, x, graph, e, mask, ground_truth, 
            self.nodes_num_list, self.raw_data_list
        )
        
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
                dists=dists[idx], ref_tour=ref_tours[idx] if ref_tours is not None else None
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
            
        # merge
        return self.merge_process(
            task="ATSP", with_gt=True if ref_tours is not None else False
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
                depot=depots[idx], points=points[idx], demand=demands[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.merge_process(
            task="CVRP", with_gt=True if ref_tours is not None else False
        )
    
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
                points=points[idx], ref_tour=ref_tours[idx] if ref_tours is not None else None, 
            )
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.merge_process(
            task="TSP", with_gt=True if ref_tours is not None else False
        )
    
    #################################
    #          Random Mask          #
    #################################
        
    def atsp_batch_random_mask(
        self, mask: Tensor, delta: float, raw_data_list: list
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        for idx in range(len(mask)):
            _random_mask = atsp_random_mask_dense(
                mask=mask[idx], delta=delta, raw_data=raw_data_list[idx]
            )
            random_mask[idx] = _random_mask
        return random_mask
    
    def cvrp_batch_random_mask(
        self, mask: Tensor, delta: float, raw_data_list: list
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        for idx in range(len(mask)):
            _random_mask = cvrp_random_mask_dense(
                mask=mask[idx], delta=delta, raw_data=raw_data_list[idx]
            )
            random_mask[idx] = _random_mask
        return random_mask
    
    def tsp_batch_random_mask(self, mask: Tensor, delta: float) -> Tensor:
        random_mask = torch.zeros_like(mask)
        for idx in range(len(mask)):
            _random_mask = tsp_random_mask_dense(mask=mask[idx], delta=delta)
            random_mask[idx] = _random_mask
        return random_mask

    #################################
    #          Update Mask          #
    #################################

    def atsp_batch_update_mask(
        self, heatmap: Tensor, e: Tensor, mask: Tensor, 
        max_steps: int, continue_flag: bool
    ) -> Tensor:
        samples_num = len(heatmap)
        for idx in range(samples_num):
            e[idx], mask[idx] = atsp_update_mask_dense(
                heatmap=heatmap[idx], e=e[idx], mask=mask[idx],
                max_steps=max_steps, continue_flag=continue_flag
            )
        return e, mask
    
    def cvrp_batch_update_mask(
        self, heatmap: Tensor, e: Tensor, mask: Tensor, raw_data_list: list,
        max_steps: int, continue_flag: bool
    ) -> Any:
        samples_num = len(heatmap)
        for idx in range(samples_num):
            e[idx], mask[idx] = cvrp_update_mask_dense(
                heatmap=heatmap[idx], e=e[idx], mask=mask[idx], 
                raw_data=raw_data_list[idx], max_steps=max_steps, 
                continue_flag=continue_flag
            )
        return e, mask
    
    def tsp_batch_update_mask(
        self, heatmap: Tensor, e: Tensor, mask: Tensor, 
        max_steps: int, continue_flag: bool
    ) -> Tensor:
        samples_num = len(heatmap)
        for idx in range(samples_num):
            e[idx], mask[idx] = tsp_update_mask_dense(
                heatmap=heatmap[idx], e=e[idx], mask=mask[idx],
                max_steps=max_steps, continue_flag=continue_flag
            )
        return e, mask