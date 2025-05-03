import torch
import numpy as np
from typing import List, Union, Any
from ml4co_kit import (
    check_dim, MClGraphData, MCutGraphData, 
    MISGraphData, MVCGraphData
)
from goal.env.data_process import (
    atsp_data_process, cvrp_data_process, 
    mcl_data_process, mcut_data_process, 
    mis_data_process, mvc_data_process, tsp_data_process
)


class GOALDataProcessor(object):
    def __init__(self, device: str) -> None:
        self.device = device

    def initial_lists(self):
        self.x_list = list()
        self.e_list = list()
        self.raw_data_list = list()
        
    def update_lists(self, data: Any):
        self.x_list.append(data[0])
        self.e_list.append(data[1])
        self.raw_data_list.append(data[2])
    
    def merge_process(self, task: str) -> Any:
        # nodes feature
        if self.x_list[0] is not None:
            x = torch.stack(self.x_list, 0)
            x = x.to(self.device)
        else:
            x = None
        
        # edges feature
        if self.e_list[0] is not None:
            e = torch.stack(self.e_list, 0)
            e = e.to(self.device)
        else:
            e = None
        
        return (task, x, e, self.raw_data_list)
    
    def atsp_batch_data_process(
        self, dists: np.ndarray, ref_tours: np.ndarray, 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # check dimension
        check_dim(dists, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for idx in range(dists.shape[0]):
            sparse_data = atsp_data_process(
                dists=dists[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
                mode=mode
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
            
        # merge
        return self.merge_process(task="ATSP")
           
    def cvrp_batch_data_process(
        self, 
        depots: Union[np.ndarray, list, List[np.ndarray]], 
        points: Union[np.ndarray, list, List[np.ndarray]], 
        demands: Union[np.ndarray, list, List[np.ndarray]], 
        ref_tours: Union[np.ndarray, list, List[np.ndarray]], 
        sampling_num: int = 1,
        mode: str = "train"
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for idx in range(points.shape[0]):
            sparse_data = cvrp_data_process(
                depot=depots[idx], 
                points=points[idx], 
                demand=demands[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
                mode=mode
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
        
        # merge
        return self.merge_process(task="CVRP")

    def mcl_batch_data_process(
        self, graph_data: List[MClGraphData], 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # max nodes
        nodes_num_list = [g.nodes_num for g in graph_data]
        max_nn = np.max(np.array(nodes_num_list))
        
        # sparse process
        for graph in graph_data:
            sparse_data = mcl_data_process(graph, max_nn)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MCl")
    
    def mcut_batch_data_process(
        self, graph_data: List[MCutGraphData], 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for graph in graph_data:
            sparse_data = mcut_data_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MCut")

    def mis_batch_data_process(
        self, graph_data: List[MISGraphData], 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # initialize lists
        self.initial_lists()
        
        # max nodes
        nodes_num_list = [g.nodes_num for g in graph_data]
        max_nn = np.max(np.array(nodes_num_list))
        
        # sparse process
        for graph in graph_data:
            sparse_data = mis_data_process(graph, max_nn)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MIS")

    def mvc_batch_data_process(
        self, graph_data: List[MVCGraphData], 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # max nodes
        nodes_num_list = [g.nodes_num for g in graph_data]
        max_nn = np.max(np.array(nodes_num_list))
        
        # sparse process
        for graph in graph_data:
            sparse_data = mvc_data_process(graph, max_nn)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process("MVC")
    
    def tsp_batch_data_process(
        self, points: np.ndarray, ref_tours: np.ndarray, 
        sampling_num: int = 1, mode: str = "train"
    ) -> Any:
        # check dimension
        check_dim(points, 3)
        check_dim(ref_tours, 2)
        
        # initialize lists
        self.initial_lists()
        
        # sparse process
        for idx in range(points.shape[0]):
            sparse_data = tsp_data_process(
                points=points[idx], 
                ref_tour=ref_tours[idx] if ref_tours is not None else None, 
                mode=mode
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
        
        # merge
        return self.merge_process(task="TSP")