import torch
from typing import Any
from torch import Tensor, nn
from ml4co_kit import BaseModel
from typing import Union, Tuple, List
from gnn4co.env.env import GNN4COEnv
from gnn4co.model.decoder.base import GNN4CODecoder
from gnn4co.model.encoder.gnn_encoder import GNNEncoder


class GNN4COModel(BaseModel):
    def __init__(
        self,
        env: GNN4COEnv,
        encoder: GNNEncoder,
        decoder: GNN4CODecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None
    ):
        super(GNN4COModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: GNN4COEnv
        self.model: GNNEncoder
        self.decoder: GNN4CODecoder = decoder
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        
        # get real data
        """
        task: ATSP or CVRP or MCl or MCut or MIS or MVC or TSP
        if sparse:
            [0] task
            [1] x: (V, C) or (V,) , nodes feature
            [2] e: (E, D) or (E,) , edges feature
            [3] edge_index: (2, E)
            [4] graph_list: graph data
            [5] ground_truth: (E,) or (V,)
            [6] nodes_num_list
            [7] edges_num_list
        else:
            [0] task
            [1] x: (B, V, C) or (B, V), nodes_feature
            [2] graph: (B, V, V)
            [3] ground_truth: (B, V, V) or (B, V)
            [4] nodes_num_list
        """
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            batch_data = self.env.generate_train_data(batch_size)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss = self.train_edge_sparse_process(*batch_data)
                else:
                    loss = self.train_edge_dense_process(*batch_data)
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss = self.train_node_sparse_process(*batch_data)
                else:
                    loss = self.train_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # get val data
            batch_data = self.env.generate_val_data(batch_idx)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_edge_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_edge_dense_process(*batch_data)
                    
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_node_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
            # decoding
            if self.env.sparse:
                costs_avg = self.decoder.sparse_decode(heatmap, *batch_data, return_cost=True)
            else:
                costs_avg = self.decoder.dense_decode(heatmap, *batch_data, return_cost=True)
        else:
            raise NotImplementedError()
     
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            formatted_v = f"{v:.8f}"
            self.log(k, float(formatted_v), prog_bar=True, on_epoch=True, sync_dist=True)
        
        # return
        return loss if phase == "train" else metrics   
        
    def train_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
   
    def train_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=graph, edge_index=None
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
    
    def train_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del e_pred
        loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
        return loss

    def train_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        raise NotImplementedError()

    def inference_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del x_pred
        
        # heatmap
        e_pred_softmax = e_pred.softmax(dim=-1)
        e_heatmap = e_pred_softmax[:, 1]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=graph, edge_index=None
        )
        del x_pred
        
        # heatmap
        e_pred_softmax = e_pred.softmax(dim=1)
        e_heatmap = e_pred_softmax[:, 1, :, :]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del e_pred
        
        # heatmap
        x_pred_softmax = x_pred.softmax(-1)
        x_heatmap = x_pred_softmax[:, 1]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
            return loss, x_heatmap
        elif self.env.mode == "solve":
            return x_heatmap
        else:
            raise ValueError()
        
    def inference_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError()