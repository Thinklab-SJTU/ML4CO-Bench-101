import torch
from typing import Any
from torch import Tensor, nn
from ml4co_kit import BaseModel
from typing import Union, Tuple, List
from vag_co.env.env import VAGCOEnv
from vag_co.model.decoder.base import VAGCODecoder
from vag_co.model.encoder.gnn_encoder import GNNEncoder


class VAGCOModel(BaseModel):
    def __init__(
        self,
        env: VAGCOEnv,
        encoder: GNNEncoder,
        decoder: VAGCODecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        penalty_coeff: float = 1.01,
        weight_path: str = None
    ):
        super(VAGCOModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: VAGCOEnv
        self.model: GNNEncoder
        self.decoder: VAGCODecoder = decoder
        self.penalty_coeff = penalty_coeff
        
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
        task: MCl or MCut or MIS or MVC
        [0] task
        [1] x: (V, C) or (V,) , nodes feature
        [2] e: (E, D) or (E,) , edges feature
        [3] edge_index: (2, E)
        [4] graph_list: graph data
        [5] ground_truth: (E,) or (V,)
        [6] nodes_num_list
        [7] edges_num_list
        """
        if phase == "train":
            batch_size = len(batch)
            batch_data = self.env.generate_train_data(batch_size)
            loss = self.train_process(*batch_data)
            
        elif phase == "val":
            batch_data = self.env.generate_val_data(batch_idx)
            loss, heatmap = self.inference_process(*batch_data)
            costs_avg = self.decoder.decode(heatmap, *batch_data, return_cost=True)
        
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
        
    def train_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, graph_list: List[Tensor], 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        # forward
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del e_pred
        
        # heatmap
        x_pred_softmax = x_pred.softmax(-1)
        x_heatmap = x_pred_softmax[:, 1]
        
        # energy
        if task == "MIS":
            # benefit
            total_benefit = torch.sum(x_heatmap)
            
            # penalty
            penalty_list = list()
            begin_nodes_num = 0
            for nodes_num, graph in zip(nodes_num_list, graph_list):
                end_nodes_num = begin_nodes_num + nodes_num
                cur_heatmap = x_heatmap[begin_nodes_num:end_nodes_num]
                graph = graph.to(self.env.device).float()
                penalty = torch.sum((torch.matmul(cur_heatmap, graph) * cur_heatmap.unsqueeze(dim=0)))
                penalty_list.append(penalty)
                begin_nodes_num = end_nodes_num
            total_penalty = torch.sum(torch.stack(penalty_list))
            
            # energy
            energy_loss = -total_benefit + self.penalty_coeff * total_penalty
            energy_loss = energy_loss / len(nodes_num_list)
            
        elif task == "MCl":
            # benefit
            total_benefit = torch.sum(x_heatmap)
            
            # penalty
            penalty_list = list()
            begin_nodes_num = 0
            for nodes_num, graph in zip(nodes_num_list, graph_list):
                end_nodes_num = begin_nodes_num + nodes_num
                cur_heatmap = x_heatmap[begin_nodes_num:end_nodes_num]
                graph = graph.to(self.env.device).float()
                comp_graph = torch.ones_like(graph) - graph
                penalty = torch.sum((torch.matmul(cur_heatmap, comp_graph) * cur_heatmap.unsqueeze(dim=0)))
                penalty_list.append(penalty)
                begin_nodes_num = end_nodes_num
            total_penalty = torch.sum(torch.stack(penalty_list))
            
            # energy
            energy_loss = -total_benefit + self.penalty_coeff * total_penalty
            energy_loss = energy_loss / len(nodes_num_list)
        
        elif task == "MVC":
            # benefit
            total_benefit = -torch.sum(x_heatmap)
            
            # penalty
            penalty_list = list()
            begin_nodes_num = 0
            for nodes_num, graph in zip(nodes_num_list, graph_list):
                end_nodes_num = begin_nodes_num + nodes_num
                cur_heatmap = x_heatmap[begin_nodes_num:end_nodes_num]
                graph = graph.to(self.env.device).float()
                minus_heatmap = torch.ones_like(cur_heatmap) - cur_heatmap
                penalty = torch.sum((torch.matmul(minus_heatmap, graph) * minus_heatmap.unsqueeze(dim=0)))
                penalty_list.append(penalty)
                begin_nodes_num = end_nodes_num
            total_penalty = torch.sum(torch.stack(penalty_list))
            
            # energy
            energy_loss = -total_benefit + self.penalty_coeff * total_penalty
            energy_loss = energy_loss / len(nodes_num_list) / 100

        elif task == "MCut":
            # benefit
            edge_index_0 = 2 * x_heatmap[edge_index[0]] - 1
            edge_index_1 = 2 * x_heatmap[edge_index[1]] - 1
            total_benefit = torch.sum(edge_index_0 * edge_index_1, dim=0)
            
            # energy
            energy_loss = total_benefit
            energy_loss = energy_loss / len(nodes_num_list) / 1000
            
        else:
            raise NotImplementedError()
        
        return energy_loss
    
    def inference_process(
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