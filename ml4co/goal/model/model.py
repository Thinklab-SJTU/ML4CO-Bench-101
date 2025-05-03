import copy
import torch
from typing import Any
from torch import Tensor, nn
from ml4co_kit import BaseModel
from typing import Union, Tuple
from goal.env.env import GOALEnv
from goal.model.encoder.gat_encoder import GATEncoder
from goal.model.decoder.decoder import GOALDecoder


class GOALModel(BaseModel):
    def __init__(
        self,
        env: GOALEnv,
        encoder: GATEncoder,
        decoder: GOALDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        beam_size: int = -1,
        weight_path: str = None,
    ):
        super(GOALModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: GOALEnv
        self.model: GATEncoder
        self.decoder = decoder
        self.beam_size = beam_size
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        
        # get real data
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            batch_data = self.env.generate_train_data(batch_size)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                loss = self.train_edge_process(*batch_data)
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                loss = self.train_node_process(*batch_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # get val data
            batch_data = self.env.generate_val_data(batch_idx)
            
            # decoding
            costs_avg = self.decoder.decode(
                *batch_data, self.model, self.beam_size, return_cost=True, 
            )
            loss = 0 # do not need loss
        else:
            raise NotImplementedError()
     
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # return
        return loss if phase == "train" else metrics   
        
    def train_edge_process(
        self, task: str, x: Tensor, e: Tensor, raw_data: Any
    ) -> Tensor:
        if task in ["ATSP", "TSP"]:
            # sub problem
            batch_size, nodes_num, _, _ = e.shape
            sub_size = torch.randint(low=4, high=nodes_num+1, size=(1,))[0]
            begin_idx = torch.randint(low=0, high=nodes_num-sub_size+1, size=(1,))[0]
            sub_e = e[:, begin_idx:begin_idx+sub_size, begin_idx:begin_idx+sub_size, :]
            
            # ground_truth
            ground_truth = torch.ones((batch_size,)).long().to(self.device)
            
            # forward
            output_scores = self.model.forward(x=x, e=sub_e, mask=None)

            # loss
            loss = nn.CrossEntropyLoss()(output_scores, ground_truth)
            return loss
        
        elif task in ["CVRP"]:
            # sub problem
            batch_size, nodes_num, _, _ = e.shape
            begin_idx = torch.randint(low=0, high=nodes_num-10, size=(1,))[0]
            sub_x = x[:, begin_idx:]
            sub_e = e[:, begin_idx:, begin_idx:]
            sub_demands = x[:, begin_idx:, 0]
            nodes_feature = copy.deepcopy(sub_x[:, :, :2])
            nodes_feature[:, 1:, 1] = 0.0
            
            # ground_truth
            ground_truth = (2 + sub_x[:, 0, 2]).long()
            
            # forward
            output_scores = self.model.forward(x=nodes_feature, e=sub_e, mask=None)
            output_scores = output_scores.reshape(batch_size, -1, 2)
            remain_cp = nodes_feature[:, 0, 1]
            output_scores[..., 0][sub_demands > remain_cp.unsqueeze(-1)] = -torch.inf
            output_scores = output_scores.reshape(batch_size, -1)
            
            # loss
            loss = nn.CrossEntropyLoss()(output_scores, ground_truth)
            return loss
            
        else:
            raise NotImplementedError()
    
    def train_node_process(
        self, task: str, x: Tensor, e: Tensor, raw_data: Any
    ) -> Tensor:        
        # forward
        output_scores = self.model.forward(x=None, e=e, mask=None)
        
        # loss
        loss = nn.BCEWithLogitsLoss()(output_scores, x.float())
        return loss