import numpy as np
from torch import Tensor
from typing import Union, List
from co_expander.model.decoder.decode.atsp import atsp_dense_decode
from co_expander.model.decoder.decode.cvrp import cvrp_dense_decode
from co_expander.model.decoder.decode.mcl import mcl_sparse_decode
from co_expander.model.decoder.decode.mcut import mcut_sparse_decode
from co_expander.model.decoder.decode.mis import mis_sparse_decode
from co_expander.model.decoder.decode.mvc import mvc_sparse_decode
from co_expander.model.decoder.decode.tsp import tsp_dense_decode, tsp_sparse_decode


class COExpanderDecoder(object):
    def __init__(self, decode_kwargs: dict = {}) -> None:
        self.decode_kwargs = decode_kwargs

    def sparse_decode(
        self, vars: Tensor, task: str, nodes_feature: Tensor, x: Tensor, 
        edges_feature: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, edges_num_list: list, 
        raw_data_list: list, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        if task == "MCl":
            return mcl_sparse_decode(
                vars=vars, graph_list=graph_list, nodes_num_list=nodes_num_list, 
                return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "MCut":
            return mcut_sparse_decode(
                vars=vars, edge_index=edge_index, graph_list=graph_list,
                nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "MIS":
            return mis_sparse_decode(
                vars=vars, graph_list=graph_list, nodes_num_list=nodes_num_list, 
                return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "MVC":
            return mvc_sparse_decode(
                vars=vars, graph_list=graph_list, nodes_num_list=nodes_num_list, 
                return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "TSP":
            return tsp_sparse_decode(
                vars=vars, nodes_feature=nodes_feature, edge_index=edge_index,
                nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                return_cost=return_cost, **self.decode_kwargs
            )
        else:
            raise NotImplementedError()
        
    def dense_decode(
        self, vars: Tensor, task: str, nodes_feature: Tensor, x: Tensor, 
        graph: Tensor, e: Tensor, mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, raw_data_list: list, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        if task == "ATSP":
            return atsp_dense_decode(
                vars=vars, graph=graph, return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "CVRP":
            return cvrp_dense_decode(
                vars=vars, nodes_feature=nodes_feature, 
                return_cost=return_cost, **self.decode_kwargs
            )
        elif task == "TSP":
            return tsp_dense_decode(
                vars=vars, nodes_feature=nodes_feature, 
                return_cost=return_cost, **self.decode_kwargs
            )
        else:
            raise NotImplementedError()