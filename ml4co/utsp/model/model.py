import torch
from typing import Any
from torch import Tensor
from ml4co_kit import BaseModel
from typing import Union, Tuple
from utsp.env.env import UTSPEnv
from utsp.model.decoder.tsp import UTSPDecoder
from utsp.model.encoder.sag_encoder import SAGEncoder


class UTSPModel(BaseModel):
    def __init__(
        self,
        env: UTSPEnv,
        encoder: SAGEncoder,
        decoder: UTSPDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None,
        tau: float = 3.5,
        penalty_loss: float = 10.0,
        diag_loss: float = 0.1,
    ):
        super(UTSPModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: UTSPEnv
        self.model: SAGEncoder
        self.decoder: UTSPDecoder = decoder
        
        # utsp args
        self.tau = tau
        self.penalty_loss = penalty_loss
        self.diag_loss = diag_loss
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            x, graph = self.env.generate_train_data(batch_size) 
            loss = self.train_process(x=x, graph=graph)

        elif phase == "val":
            # get val data
            x, graph = self.env.generate_val_data(batch_idx)
            
            # deal with different task
            loss, heatmap = self.inference_process(x=x, graph=graph)
            costs_avg = self.decoder.decode(heatmap=heatmap, x=x, return_cost=True)

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
    
    def utsp_loss(self, output: Tensor, heatmap: Tensor, graph: Tensor) -> Tensor:
        # get heatmap and weighted path
        weighted_path = torch.mul(heatmap, graph).sum(dim=(1, 2))
        
        # heatmap diagonals and normalize
        heatmap_diagonals = [torch.diagonal(mat) for mat in heatmap]
        heatmap_diagonals = torch.stack(heatmap_diagonals, dim=0)
        nrmlzd_constraint = (1. - torch.sum(output,2))**2
        nrmlzd_constraint = torch.sum(nrmlzd_constraint)
        
        # utsp loss
        loss_term1 = self.penalty_loss * nrmlzd_constraint
        loss_term2 = 1.0 * torch.sum(weighted_path)
        loss_term3 = self.diag_loss * torch.sum(heatmap_diagonals)
        batch_loss = torch.sum(loss_term1 + loss_term2 + loss_term3)
        
        return batch_loss
    
    def utsp_heatmap(self, output: Tensor) -> Tensor:
        heatmap = torch.matmul(output, torch.roll(torch.transpose(output, 1, 2),-1, 1))
        return heatmap
        
    def train_process(self, x: Tensor, graph: Tensor) -> Tensor:
        # forward
        tau_adj = torch.exp(-1.0 * graph / self.tau)
        diagonal_indices = torch.arange(tau_adj.size(1))
        tau_adj[:, diagonal_indices, diagonal_indices] = 0
        output = self.model.forward(x=x, graph=tau_adj)
        
        # heatmap
        heatmap = self.utsp_heatmap(output)
        
        # get loss
        batch_loss = self.utsp_loss(output, heatmap, graph)
        loss = batch_loss / len(x)
        
        return loss
    
    def inference_process(self, x: Tensor, graph: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        tau_adj = torch.exp(-1.0 * graph / self.tau)
        diagonal_indices = torch.arange(tau_adj.size(1))
        tau_adj[:, diagonal_indices, diagonal_indices] = 0
        output = self.model.forward(x=x, graph=tau_adj)
        
        # heatmap
        heatmap = self.utsp_heatmap(output)
        
        # return
        if self.env.mode == "val":
            batch_loss = self.utsp_loss(output, heatmap, graph)
            loss = batch_loss / len(x)
            return loss, heatmap
        elif self.env.mode == "solve":
            return heatmap
        else:
            raise ValueError()