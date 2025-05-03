import torch
import numpy as np
from typing import List, Union, Any
from ml4co_kit import (
    check_dim, MClGraphData, MCutGraphData, 
    MISGraphData, MVCGraphData
)
from gnn4co.env.sparse import (
    atsp_sparse_process, cvrp_sparse_process, 
    mcl_sparse_process, mcut_sparse_process, 
    mis_sparse_process, mvc_sparse_process, tsp_sparse_process
)


class GNN4COSparser(object):
    def __init__(self, sparse_factor: int, device: str) -> None:
        self.sparse_factor = sparse_factor
        self.device = device
    
    def initial_lists(self):
        self.x_list = list()
        self.e_list = list()
        self.edge_index_list = list()
        self.graph_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        self.edges_num_list = list()
        
    def update_lists(self, sparse_data: Any):
        self.x_list.append(sparse_data[0])
        self.e_list.append(sparse_data[1])
        self.edge_index_list.append(sparse_data[2])
        self.graph_list.append(sparse_data[3])
        self.ground_truth_list.append(sparse_data[4])
        self.nodes_num_list.append(sparse_data[5])
        self.edges_num_list.append(sparse_data[6])
    
    def merge_process(self, task: str, with_gt: bool) -> Any:
        # nodes feature
        if self.x_list[0] is not None:
            x = torch.cat(self.x_list, 0).to(self.device) # (V, C) or (V,)
        else:
            x = None
            
        # edges feature
        if self.e_list[0] is not None:
            e = torch.cat(self.e_list, 0).to(self.device) # (V, C) or (E,)
        else:
            e = None

        # edge index
        add_index = 0
        edge_index_list = list()
        for idx, edge_index in enumerate(self.edge_index_list):
            edge_index_list.append(edge_index + add_index)
            add_index += self.nodes_num_list[idx]
        edge_index = torch.cat(edge_index_list, 1).to(self.device) # (2, E)

        # ground truth
        if with_gt:
            ground_truth = torch.cat(self.ground_truth_list, 0).to(self.device) # (E,) or (V,)
        else:
            ground_truth = None
            
        return (
            task, x, e, edge_index, self.graph_list, 
            ground_truth, self.nodes_num_list, self.edges_num_list
        )
    
    def atsp_batch_data_process(
        self, dists: np.ndarray, ref_tours: np.ndarray, sampling_num: int = 1
    ) -> Any:
        # check dimension
        check_dim(dists, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for idx in range(dists.shape[0]):
            sparse_data = atsp_sparse_process(
                dists=dists[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
                sparse_factor=self.sparse_factor
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
            
        # merge
        return self.merge_process(
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
        
        # sparse process
        for idx in range(points.shape[0]):
            sparse_data = cvrp_sparse_process(
                depot=depots[idx], 
                points=points[idx], 
                demand=demands[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
                sparse_factor=self.sparse_factor
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
        
        # merge
        return self.merge_process(
            task="CVRP", 
            with_gt=True if ref_tours is not None else False
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
        return self.merge_process("MCl", True)
    
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
        return self.merge_process("MCut", True)

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
        return self.merge_process("MIS", True)

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
        return self.merge_process("MVC", True)
    
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
            task="TSP", 
            with_gt=True if ref_tours is not None else False
        )