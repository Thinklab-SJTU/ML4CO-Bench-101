import torch
from typing import List, Any
from ml4co_kit import (
    MClGraphData, MCutGraphData, MISGraphData, MVCGraphData
)
from vag_co.env.sparse import (
    mcl_sparse_process, mcut_sparse_process, mis_sparse_process, mvc_sparse_process
)


class VAGCOSparser(object):
    def __init__(self, device: str) -> None:
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