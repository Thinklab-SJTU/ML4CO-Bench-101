import torch
import numpy as np
from torch import Tensor
from typing import List, Union, Any
from ml4co_kit import (
    check_dim, MClGraphData, MCutGraphData, 
    MISGraphData, MVCGraphData
)
from co_expander.env.processor import DataProcessor
from co_expander.env.sparse import (
    mcl_sparse_process, mcut_sparse_process, 
    mis_sparse_process, mvc_sparse_process, tsp_sparse_process
)
from co_expander.env.mask import (
    mcl_random_mask_sparse, mcl_update_mask_sparse,
    mcut_random_mask_sparse, mcut_update_mask_sparse, 
    mis_random_mask_sparse, mis_update_mask_sparse,
    mvc_random_mask_sparse, mvc_update_mask_sparse,    
    tsp_random_mask_sparse, tsp_update_mask_sparse,
)


class COExpanderSparser(DataProcessor):
    def __init__(self, sparse_factor: int, device: str) -> None:
        super().__init__(device=device)
        self.sparse_factor = sparse_factor
        
    #################################
    #        Raw Data Process       #
    #################################
    
    def initial_lists(self):
        self.nodes_feature_list = list()
        self.x_list = list()
        self.edges_feature_list = list()
        self.e_list = list()
        self.edge_index_list = list()
        self.graph_list = list()
        self.mask_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        self.edges_num_list = list()
        self.raw_data_list = list()
        
    def update_lists(self, sparse_data: Any):
        self.nodes_feature_list.append(sparse_data[0])
        self.x_list.append(sparse_data[1])
        self.edges_feature_list.append(sparse_data[2])
        self.e_list.append(sparse_data[3])
        self.edge_index_list.append(sparse_data[4])
        self.graph_list.append(sparse_data[5])
        self.mask_list.append(sparse_data[6])
        self.ground_truth_list.append(sparse_data[7])
        self.nodes_num_list.append(sparse_data[8])
        self.edges_num_list.append(sparse_data[9])
        self.raw_data_list.append(sparse_data[10])
    
    def merge_process(self, task: str, with_gt: bool) -> Any:
        # [0] nodes' feature
        if self.nodes_feature_list[0] is not None:
            nodes_feature = torch.cat(self.nodes_feature_list, 0) # (V, C) or (V,)
            nodes_feature = nodes_feature.to(self.device)
        else:
            nodes_feature = None
        
        # [1] nodes' decision variable
        if self.x_list[0] is not None:
            x = torch.cat(self.x_list, 0).to(self.device) # (V,)
        else:
            x = None
        
        # [2] edges' feature
        if self.edges_feature_list[0] is not None:
            edges_feature = torch.cat(self.edges_feature_list, 0) # (V, C) or (E,)
            edges_feature = edges_feature.to(self.device)
        else:
            edges_feature = None
            
        # [3] edges' decision variable
        if self.e_list[0] is not None:
            e = torch.cat(self.e_list, 0).to(self.device) # (E,)
        else:
            e = None
          
        # [4] edge index
        add_index = 0
        edge_index_list = list()
        for idx, edge_index in enumerate(self.edge_index_list):
            edge_index_list.append(edge_index + add_index)
            add_index += self.nodes_num_list[idx]
        edge_index = torch.cat(edge_index_list, 1).to(self.device) # (2, E)
        
        # [6] mask
        mask = torch.cat(self.mask_list, 0).to(self.device) # (V,) or (E,)

        # [7] ground truth
        if with_gt:
            ground_truth = torch.cat(self.ground_truth_list, 0).to(self.device) # (E,) or (V,)
        else:
            ground_truth = None
        
        # return processed dense data (12 items)
        return (
            task, nodes_feature, x, edges_feature, e, edge_index, self.graph_list, 
            mask, ground_truth, self.nodes_num_list, self.edges_num_list, self.raw_data_list
        )

    def mcl_batch_data_process(
        self, graph_data: List[MClGraphData], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for graph in graph_data:
            sparse_data = mcl_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MCl", with_gt=True)
    
    def mcut_batch_data_process(
        self, graph_data: List[MCutGraphData], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for graph in graph_data:
            sparse_data = mcut_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MCut", with_gt=True)

    def mis_batch_data_process(
        self, graph_data: List[MISGraphData], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for graph in graph_data:
            sparse_data = mis_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MIS", with_gt=True)

    def mvc_batch_data_process(
        self, graph_data: List[MVCGraphData], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for graph in graph_data:
            sparse_data = mvc_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MVC", with_gt=True)
    
    def tsp_batch_data_process(
        self, points: np.ndarray, ref_tours: np.ndarray, sampling_num: int = 1
    ) -> Any:
        # check dimension
        check_dim(points, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for idx in range(points.shape[0]):
            sparse_data = tsp_sparse_process(
                points=points[idx], 
                ref_tour=ref_tours[idx] if ref_tours  is not None else None, 
                sparse_factor=self.sparse_factor
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
        
        # merge
        return self.merge_process(
            task="TSP", with_gt=True if ref_tours is not None else False
        )

    #################################
    #          Random Mask          #
    #################################
    
    def mcl_batch_random_mask(
        self, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, delta: float
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        begin_idx = 0
        for idx in range(len(nodes_num_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            _random_mask = mcl_random_mask_sparse(
                graph=graph_list[idx], mask=mask[begin_idx:end_idx],
                ground_truth=ground_truth[begin_idx:end_idx], delta=delta
            )
            random_mask[begin_idx:end_idx] = _random_mask
            begin_idx = end_idx
        return random_mask
    
    def mcut_batch_random_mask(
        self, mask: Tensor, nodes_num_list: list, delta: float
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        begin_idx = 0
        for idx in range(len(nodes_num_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            _random_mask = mcut_random_mask_sparse(
                mask=mask[begin_idx:end_idx], delta=delta
            )
            random_mask[begin_idx:end_idx] = _random_mask
            begin_idx = end_idx
        return random_mask
    
    def mis_batch_random_mask(
        self, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, delta: float
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        begin_idx = 0
        for idx in range(len(nodes_num_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            _random_mask = mis_random_mask_sparse(
                graph=graph_list[idx], mask=mask[begin_idx:end_idx],
                ground_truth=ground_truth[begin_idx:end_idx], delta=delta
            )
            random_mask[begin_idx:end_idx] = _random_mask
            begin_idx = end_idx
        return random_mask
    
    def mvc_batch_random_mask(
        self, mask: Tensor, nodes_num_list: list, delta: float
    ) -> Tensor:
        random_mask = torch.zeros_like(mask)
        begin_idx = 0
        for idx in range(len(nodes_num_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            _random_mask = mvc_random_mask_sparse(
                mask=mask[begin_idx:end_idx], delta=delta
            )
            random_mask[begin_idx:end_idx] = _random_mask
            begin_idx = end_idx
        return random_mask
    
    def tsp_batch_random_mask(
        self, mask: Tensor, edges_num_list: list, delta: float
    ) -> Any:
        random_mask = torch.zeros_like(mask)
        begin_idx = 0
        for idx in range(len(edges_num_list)):
            end_idx = begin_idx + edges_num_list[idx]
            _random_mask = tsp_random_mask_sparse(
                mask=mask[begin_idx:end_idx], delta=delta
            )
            random_mask[begin_idx:end_idx] = _random_mask
            begin_idx = end_idx
        return random_mask

    #################################
    #          Update Mask          #
    #################################

    def mcl_batch_update_mask(
        self, heatmap: Tensor, x: Tensor, graph_list: List[Tensor], mask: Tensor, 
        nodes_num_list: list, max_steps: int, continue_flag: bool, beam_size: int = -1
    ) -> Tensor:
        begin_idx = 0
        for idx in range(len(graph_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            x[begin_idx:end_idx], mask[begin_idx:end_idx] = mcl_update_mask_sparse(
                heatmap=heatmap[begin_idx:end_idx], x=x[begin_idx:end_idx], 
                graph=graph_list[idx], mask=mask[begin_idx:end_idx],
                max_steps=max_steps, continue_flag=continue_flag, beam_size=beam_size
            )
            begin_idx = end_idx
        return x, mask
    
    def mcut_batch_update_mask(
        self, heatmap: Tensor, x: Tensor, graph_list: List[Tensor], 
        mask: Tensor, nodes_num_list: list, max_steps: int, continue_flag: bool
    ) -> Tensor:
        begin_idx = 0
        for idx in range(len(graph_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            x[begin_idx:end_idx], mask[begin_idx:end_idx] = mcut_update_mask_sparse(
                heatmap=heatmap[begin_idx:end_idx], x=x[begin_idx:end_idx], 
                graph=graph_list[idx], mask=mask[begin_idx:end_idx],
                max_steps=max_steps, continue_flag=continue_flag
            )
            begin_idx = end_idx
        return x, mask
    
    def mis_batch_update_mask(
        self, heatmap: Tensor, x: Tensor, graph_list: List[Tensor], mask: Tensor, 
        nodes_num_list: list, max_steps: int, continue_flag: bool, beam_size: int = -1
    ) -> Tensor:
        begin_idx = 0
        for idx in range(len(graph_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            x[begin_idx:end_idx], mask[begin_idx:end_idx] = mis_update_mask_sparse(
                heatmap=heatmap[begin_idx:end_idx], 
                x=x[begin_idx:end_idx], 
                graph=graph_list[idx], 
                mask=mask[begin_idx:end_idx],
                max_steps=max_steps, 
                continue_flag=continue_flag,
                beam_size=beam_size
            )
            begin_idx = end_idx
        return x, mask
    
    def mvc_batch_update_mask(
        self, heatmap: Tensor, x: Tensor, graph_list: List[Tensor], 
        mask: Tensor, nodes_num_list: list, max_steps: int, continue_flag: bool
    ) -> Tensor:
        begin_idx = 0
        for idx in range(len(graph_list)):
            end_idx = begin_idx + nodes_num_list[idx]
            x[begin_idx:end_idx], mask[begin_idx:end_idx] = mvc_update_mask_sparse(
                heatmap=heatmap[begin_idx:end_idx], x=x[begin_idx:end_idx], 
                graph=graph_list[idx], mask=mask[begin_idx:end_idx],
                max_steps=max_steps, continue_flag=continue_flag
            )
            begin_idx = end_idx
        return x, mask
    
    def tsp_batch_update_mask(
        self, heatmap: Tensor, e: Tensor, mask: Tensor, 
        edges_num_list: list, max_steps: int, continue_flag: bool
    ) -> Tensor:
        begin_idx = 0
        for idx in range(len(edges_num_list)):
            end_idx = begin_idx + edges_num_list[idx]
            e[begin_idx:end_idx], mask[begin_idx:end_idx] = tsp_update_mask_sparse(
                heatmap=heatmap[begin_idx:end_idx], e=e[begin_idx:end_idx], 
                mask=mask[begin_idx:end_idx], max_steps=max_steps, 
                continue_flag=continue_flag
            )
            
            begin_idx = end_idx
        return e, mask
    
    #################################
    #            Finetune           #
    #################################
  
    def mcut_finetune(
        pred: Tensor, edges_feature: Tensor, edge_index: Tensor
    ) -> Any:
        heatmap = pred.softmax(1)[:, 1].reshape(-1)
        heatmap = heatmap * 2 - 1
        heatmap_source = heatmap[edge_index[0]]
        heatmap_target = heatmap[edge_index[1]]
        energy = torch.sum(heatmap_source * heatmap_target * edges_feature)
        energy = energy / 1000
        return energy