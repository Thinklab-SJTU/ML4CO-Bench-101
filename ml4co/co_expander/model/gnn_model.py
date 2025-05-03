import torch
from torch import Tensor, nn
from typing import Union, Tuple, List
from co_expander.env.env import COExpanderEnv
from co_expander.model.model_base import COExpanderModel
from co_expander.model.encoder.gnn_encoder import GNNEncoder


class COExpanderGNNModel(COExpanderModel):
    def __init__(
        self,
        env: COExpanderEnv,
        encoder: GNNEncoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        beta: float = 0.025,
        prompt_prob: float = 0.1,
        delta_scale: tuple = (0, 1),
        determinate_steps: int = 1,
        beam_size: int = -1,
        weight_path: str = None,
        energy_finetune: bool = False
    ):
        super(COExpanderGNNModel, self).__init__(
            env=env,
            encoder=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            prompt_prob=prompt_prob,
            delta_scale=delta_scale,
            determinate_steps=determinate_steps,
            beam_size=beam_size,
            weight_path=weight_path,
            energy_finetune=energy_finetune
        )
        self.beta = beta
    
    def train_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Tensor:
        # random init
        x_noised = (torch.randn_like(x) > 0).long().float()
        
        # add small random noise
        x_small_noise = 1.0 + self.beta * torch.rand_like(x_noised)
        x_noised = x_noised * x_small_noise
        
        # random mask
        random_mask = self.env.random_mask_sparse(
            graph_list=graph_list, mask=mask, ground_truth=ground_truth, 
            nodes_num_list=nodes_num_list, edges_num_list=edges_num_list, 
            prompt_prob=self.prompt_prob, delta_scale=self.delta_scale
        )
        
        # update label with mask
        x_noised_mask = torch.where(random_mask, ground_truth-0.5, x_noised-0.5)

        # forward
        x_pred, e_pred = self.model.forward(
            task=task, focus_on_node=True, focus_on_edge=False,
            nodes_feature=nodes_feature, x=x_noised_mask, 
            edges_feature=edges_feature, e=e, mask=random_mask, 
            t=None, edge_index=edge_index
        )
        del e_pred
        
        # loss
        if self.energy_finetune:
            loss = self.env.finetune_sparse(x_pred, edges_feature, edge_index)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(x_pred, ground_truth)
        
        return loss

    def inference_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, 
        ground_truth: Tensor, nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # determinate_steps
        current_determinate_steps = 0
        max_steps = len(x) // self.determinate_steps
        
        # while loop, until all nodes are determined
        while not mask.all():
            # random init
            x_noised = (torch.randn_like(x) > 0).long().float()
        
            # add small random noise
            x_small_noise = 1.0 + self.beta * torch.rand_like(x_noised)
            x_noised = x_noised * x_small_noise
                
            # mask
            x_noised_mask = torch.where(mask, x-0.5, x_noised-0.5)

            # forward
            x_pred, e_pred = self.model.forward(
                task=task, focus_on_node=True, focus_on_edge=False, 
                nodes_feature=nodes_feature, x=x_noised_mask, 
                edges_feature=edges_feature, e=e, mask=mask, 
                t=None, edge_index=edge_index
            )
            del e_pred
            
            # softmax
            x_pred_softmax = x_pred.softmax(-1)
        
            # heatmap
            x_heatmap = x_pred_softmax[:, 1]

            # determine action
            if current_determinate_steps < self.determinate_steps - 1:
                x, mask = self.env.update_mask_sparse(
                    heatmap=x_heatmap, x=x, e=e, edge_index=edge_index, graph_list=graph_list, 
                    mask=mask, nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                    max_steps=max_steps, continue_flag=False, beam_size=self.beam_size
                )
            else:
                x, mask = self.env.update_mask_sparse(
                    heatmap=x_heatmap, x=x, e=e, edge_index=edge_index, graph_list=graph_list, 
                    mask=mask, nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                    max_steps=max_steps, continue_flag=True, beam_size=self.beam_size
                )
            current_determinate_steps += 1
            if current_determinate_steps == self.determinate_steps:
                break
            
        # return loss and determined variables if mode is ``val``
        if self.env.mode == "val":
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(x_pred, ground_truth)
            return loss, x
        
        # return determined variables if mode is ``solve``
        if self.env.mode == "solve":
            return x