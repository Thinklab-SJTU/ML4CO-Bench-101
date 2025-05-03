import torch
from typing import Any
from torch import Tensor
from ml4co_kit import BaseModel
from typing import Union, Tuple, List
from co_expander.env.env import COExpanderEnv
from co_expander.model.encoder.gnn_encoder import GNNEncoder
from co_expander.model.decoder.decoder import COExpanderDecoder


class COExpanderModel(BaseModel):
    def __init__(
        self,
        env: COExpanderEnv,
        encoder: GNNEncoder,
        decoder: COExpanderDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        prompt_prob: float = 0.1,
        delta_scale: tuple = (0, 1),
        determinate_steps: int = 1,
        beam_size: int = -1,
        weight_path: str = None,
        energy_finetune: bool = False
    ):
        super(COExpanderModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: COExpanderEnv
        self.model: GNNEncoder
        self.decoder = decoder
        self.delta_scale = delta_scale
        self.prompt_prob = prompt_prob
        self.determinate_steps = determinate_steps
        self.beam_size = beam_size
        self.energy_finetune = energy_finetune
        
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
            [1] nodes_feature: (V, C) or (V,) 
            [2] x: (V,), Decision variable for Nodes
            [3] edges_feature: (E, D) or (E,) 
            [4] e: (E,), Decision variable for Edges
            [5] edge_index: (2, E)
            [6] graph_list: for mask
            [7] mask: (E,) or (V,)
            [8] ground_truth: (E,) or (V,)
            [9] nodes_num_list
            [10] edges_num_list
            [11] raw_data_list
        else:
            [0] task
            [1] nodes_feature: (B, V, C) or (B, V)
            [2] x: (B, V), Decision variable for Nodes
            [3] graph: (B, V, V)
            [4] e: (B, V, V), Decision variable for Edges
            [5] mask: (B, V, V) or (B, V)
            [6] ground_truth: (B, V, V) or (B, V)
            [7] nodes_num_list
            [8] raw_data_list
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
                    loss, vars = self.inference_edge_sparse_process(*batch_data)
                else:
                    loss, vars = self.inference_edge_dense_process(*batch_data)
                    
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss, vars = self.inference_node_sparse_process(*batch_data)
                else:
                    loss, vars = self.inference_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
            # decoding
            if self.env.sparse:
                costs_avg = self.decoder.sparse_decode(vars, *batch_data, return_cost=True)
            else:
                costs_avg = self.decoder.dense_decode(vars, *batch_data, return_cost=True)
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
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Tensor:
        raise NotImplementedError(
            "``train_edge_sparse_process`` is required to implemented in subclasses."
        )
   
    def train_edge_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Tensor:
        raise NotImplementedError(
            "``train_edge_dense_process`` is required to implemented in subclasses."
        )
    
    def train_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Tensor:
        raise NotImplementedError(
            "``train_node_sparse_process`` is required to implemented in subclasses."
        )

    def train_node_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Tensor:
        raise NotImplementedError(
            "``inference_edge_sparse_process`` is required to implemented in subclasses."
        )

    def inference_edge_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError(
            "``inference_edge_sparse_process`` is required to implemented in subclasses."
        )

    def inference_edge_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError(
            "``inference_edge_dense_process`` is required to implemented in subclasses."
        )

    def inference_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError(
            "``inference_node_sparse_process`` is required to implemented in subclasses."
        )
        
    def inference_node_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError(
            "``inference_node_dense_process`` is required to implemented in subclasses."
        )