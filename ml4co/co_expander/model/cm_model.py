import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Union, Tuple, List
from co_expander.env.env import COExpanderEnv
from co_expander.model.model_base import COExpanderModel
from co_expander.model.encoder.gnn_encoder import GNNEncoder
from co_expander.model.decoder.decoder import COExpanderDecoder
from co_expander.model.consistency.inference import InferenceSchedule
from co_expander.model.consistency.diffusion import CategoricalDiffusion


class COExpanderCMModel(COExpanderModel):
    def __init__(
        self,
        env: COExpanderEnv,
        encoder: GNNEncoder,
        decoder: COExpanderDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        cm_alpha: float = 0.2,
        cm_beta: float = 0.025,
        prompt_prob: float = 0.1,
        delta_scale: tuple = (0, 1),
        inference_steps: int = 1,
        determinate_steps: int = 1,
        beam_size: int = -1,
        weight_path: str = None,
        energy_finetune: bool = False
    ):
        super(COExpanderCMModel, self).__init__(
            env=env,
            encoder=encoder,
            decoder=decoder,
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
        self.diffusion = CategoricalDiffusion(T=1000, schedule="linear")
        self.cm_alpha = cm_alpha
        self.cm_beta = cm_beta
        self.inference_steps = inference_steps
        self.time_schedule = InferenceSchedule("cosine", 1000, inference_steps)
        
    def train_edge_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Tensor:
        # consistency time
        t1: Tensor = torch.randint(1, 1001, size=(1,), device=self.device)
        t2 = (self.cm_alpha * t1).int()
        
        # diffusion (add noise based on ground truth)
        e_noised_t1, e_noised_t2 = self.diffusion.consistency_sample_sparse(
            x=ground_truth, t1=t1, t2=t2
        )

        # add small random noise
        e_small_noise_t1 = 1.0 + self.cm_beta * torch.rand_like(e_noised_t1)
        e_small_noise_t2 = 1.0 + self.cm_beta * torch.rand_like(e_noised_t2)
        e_noised_t1 = e_noised_t1 * e_small_noise_t1
        e_noised_t2 = e_noised_t2 * e_small_noise_t2
        
        # random mask
        random_mask = self.env.random_mask_sparse(
            graph_list=graph_list, mask=mask, ground_truth=ground_truth, 
            nodes_num_list=nodes_num_list, edges_num_list=edges_num_list, 
            prompt_prob=self.prompt_prob, delta_scale=self.delta_scale
        )

        # update label with mask
        e_noised_t1_mask = torch.where(random_mask, ground_truth-0.5, e_noised_t1-0.5)
        e_noised_t2_mask = torch.where(random_mask, ground_truth-0.5, e_noised_t2-0.5)

        # forward
        x_pred_t1, e_pred_t1 = self.model.forward(
            task=task, focus_on_node=False, focus_on_edge=True, 
            nodes_feature=nodes_feature, x=x, edges_feature=edges_feature, 
            e=e_noised_t1_mask, mask=random_mask, t=t1, edge_index=edge_index
        )
        x_pred_t2, e_pred_t2 = self.model.forward(
            task=task, focus_on_node=False, focus_on_edge=True, 
            nodes_feature=nodes_feature, x=x, edges_feature=edges_feature, 
            e=e_noised_t2_mask, mask=random_mask, t=t2, edge_index=edge_index
        )
        del x_pred_t1
        del x_pred_t2

        # loss
        if self.energy_finetune:
            loss_t1 = self.env.finetune_sparse(e_pred_t1, edges_feature, edge_index)
            loss_t2 = self.env.finetune_sparse(e_pred_t2, edges_feature, edge_index)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss_t1 = loss_func(e_pred_t1, ground_truth)
            loss_t2 = loss_func(e_pred_t2, ground_truth)
        loss = loss_t1 + loss_t2
        return loss
   
    def train_edge_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Tensor:
        # consistency time
        t1: Tensor = torch.randint(1, 1001, size=(1,), device=self.device)
        t2 = (self.cm_alpha * t1).int()
        
        # diffusion (add noise based on ground truth)
        e_noised_t1, e_noised_t2 = self.diffusion.consistency_sample_dense(
            x=ground_truth, t1=t1, t2=t2
        )
        
        # add small random noise
        e_small_noise_t1 = 1.0 + self.cm_beta * torch.rand_like(e_noised_t1)
        e_small_noise_t2 = 1.0 + self.cm_beta * torch.rand_like(e_noised_t2)
        e_noised_t1 = e_noised_t1 * e_small_noise_t1
        e_noised_t2 = e_noised_t2 * e_small_noise_t2
         
        # random mask
        random_mask = self.env.random_mask_dense(
            mask=mask, raw_data_list=raw_data_list,
            prompt_prob=self.prompt_prob, delta_scale=self.delta_scale
        )

        # update label with mask
        e_noised_t1_mask = torch.where(random_mask, ground_truth-0.5, e_noised_t1-0.5)
        e_noised_t2_mask = torch.where(random_mask, ground_truth-0.5, e_noised_t2-0.5)
        
        # forward
        x_pred_t1, e_pred_t1 = self.model.forward(
            task=task, focus_on_node=False, focus_on_edge=True, 
            nodes_feature=nodes_feature, x=x, edges_feature=graph, 
            e=e_noised_t1_mask, mask=random_mask, t=t1, edge_index=None
        )
        x_pred_t2, e_pred_t2 = self.model.forward(
            task=task, focus_on_node=False, focus_on_edge=True, 
            nodes_feature=nodes_feature, x=x, edges_feature=graph, 
            e=e_noised_t2_mask, mask=random_mask, t=t2, edge_index=None
        )
        del x_pred_t1
        del x_pred_t2
        
        # loss
        if self.energy_finetune:
            loss_t1 = self.env.finetune_dense(e_pred_t1, graph)
            loss_t2 = self.env.finetune_dense(e_pred_t2, graph)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss_t1 = loss_func(e_pred_t1, ground_truth)
            loss_t2 = loss_func(e_pred_t2, ground_truth)
        loss = loss_t1 + loss_t2
        return loss
    
    def train_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Tensor:
        # consistency time
        t1: Tensor = torch.randint(1, 1001, size=(1,)).to(self.device)
        t2 = (self.cm_alpha * t1).int()
        
        # diffusion (add noise based on ground truth)
        x_noised_t1, x_noised_t2 = self.diffusion.consistency_sample_sparse(
            x=ground_truth, t1=t1, t2=t2
        )
        
        # add small random noise
        x_small_noise_t1 = 1.0 + self.cm_beta * torch.rand_like(x_noised_t1)
        x_small_noise_t2 = 1.0 + self.cm_beta * torch.rand_like(x_noised_t2)
        x_noised_t1 = x_noised_t1 * x_small_noise_t1
        x_noised_t2 = x_noised_t2 * x_small_noise_t2
        
        # random mask
        random_mask = self.env.random_mask_sparse(
            graph_list=graph_list, mask=mask, ground_truth=ground_truth, 
            nodes_num_list=nodes_num_list, edges_num_list=edges_num_list, 
            prompt_prob=self.prompt_prob, delta_scale=self.delta_scale
        )
        
        # update label with mask
        x_noised_t1_mask = torch.where(random_mask, ground_truth-0.5, x_noised_t1-0.5)
        x_noised_t2_mask = torch.where(random_mask, ground_truth-0.5, x_noised_t2-0.5)
        
        # forward
        x_pred_t1, e_pred_t1 = self.model.forward(
            task=task, focus_on_node=True, focus_on_edge=False,
            nodes_feature=nodes_feature, x=x_noised_t1_mask, 
            edges_feature=edges_feature, e=e, mask=random_mask, 
            t=t1, edge_index=edge_index
        )
        x_pred_t2, e_pred_t2 = self.model.forward(
            task=task, focus_on_node=True, focus_on_edge=False,
            nodes_feature=nodes_feature, x=x_noised_t2_mask, 
            edges_feature=edges_feature, e=e, mask=random_mask, 
            t=t2, edge_index=edge_index
        )
        del e_pred_t1
        del e_pred_t2
        
        # loss
        if self.energy_finetune:
            loss_t1 = self.env.finetune_sparse(x_pred_t1, edges_feature, edge_index)
            loss_t2 = self.env.finetune_sparse(x_pred_t2, edges_feature, edge_index)
        else:
            loss_func = nn.CrossEntropyLoss()
            loss_t1 = loss_func(x_pred_t1, ground_truth)
            loss_t2 = loss_func(x_pred_t2, ground_truth)
        loss = loss_t1 + loss_t2
        
        return loss

    def inference_edge_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # determinate_steps and max_steps
        current_determinate_steps = 0
        if task == "TSP":
            max_steps_list = get_tsp_max_steps_list(
                nodes_num=nodes_num_list[0],
                determinate_steps=self.determinate_steps,
                inference_steps=self.inference_steps
            )
        else:
            raise NotImplementedError()
        
        # while loop, until all nodes are determined
        while not mask.all():
        
            # random init
            e_noised = (torch.randn_like(e) > 0).float()
            
            # denoise steps
            for i in range(self.inference_steps):
                # consistency time
                t1, t2 = self.time_schedule(i)
                t1 = torch.tensor([t1], device=self.device).float()
                
                # add small random noise
                e_small_noise = 1.0 + self.cm_beta * torch.rand_like(e_noised)
                e_noised = e_noised * e_small_noise 
                
                # mask
                e_noised_mask = torch.where(mask, e-0.5, e_noised-0.5)

                # forward
                x_pred, e_pred = self.model.forward(
                    task=task, focus_on_node=False, focus_on_edge=True, 
                    nodes_feature=nodes_feature, x=x, edges_feature=edges_feature, 
                    e=e_noised_mask, mask=mask, t=t1, edge_index=edge_index
                )
                del x_pred
                
                # softmax
                e_pred_softmax = e_pred.softmax(-1)
                
                # t2
                if t2 != 0:
                    e_pred_ber = torch.bernoulli(e_pred_softmax[..., 1].clamp(0, 1))
                    e_pred_ber_onehot: Tensor = F.one_hot(e_pred_ber.long(), num_classes=2)
                    Q_bar = torch.from_numpy(self.diffusion.Q_bar[t2]).float().to(self.device)
                    e_prob = torch.matmul(e_pred_ber_onehot.float(), Q_bar)
                    e_noised = torch.bernoulli(e_prob[..., 1].clamp(0, 1))

            # heatmap
            e_heatmap = e_pred_softmax[:, 1]
            
            # determine action
            continue_flag = not (current_determinate_steps < self.determinate_steps - 1)
            e, mask = self.env.update_mask_sparse(
                heatmap=e_heatmap, x=x, e=e, graph_list=graph_list, mask=mask, 
                nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                max_steps=max_steps_list[current_determinate_steps], 
                continue_flag=continue_flag, beam_size=self.beam_size
            )
            current_determinate_steps += 1
            if current_determinate_steps == self.determinate_steps:
                break
            
        # return loss and heatmap if mode is ``val``
        if self.env.mode == "val":
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(e_pred, ground_truth)
            return loss, e
        
        # return heatmap if mode is ``solve``
        if self.env.mode == "solve":
            return e
    
    def inference_edge_dense_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, graph: Tensor, e: Tensor, 
        mask: Tensor, ground_truth: Tensor, nodes_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # determinate_steps and max steps
        current_determinate_steps = 0
        if task == "TSP":
            max_steps_list = get_tsp_max_steps_list(
                nodes_num=nodes_num_list[0],
                determinate_steps=self.determinate_steps,
                inference_steps=self.inference_steps
            )
        elif task == "ATSP":
            max_steps_list = get_atsp_max_steps_list(
                nodes_num=nodes_num_list[0],
                determinate_steps=self.determinate_steps,
                inference_steps=self.inference_steps
            )
            mask[:, torch.arange(e.shape[-1]), torch.arange(e.shape[-1])] = True
        elif task == "CVRP":
            max_steps_list = get_cvrp_max_steps_list(
                nodes_num=nodes_num_list[0],
                determinate_steps=self.determinate_steps,
                inference_steps=self.inference_steps
            )
            mask[:, torch.arange(e.shape[-1]), torch.arange(e.shape[-1])] = True
        else:
            raise NotImplementedError()
        
        
        # while loop, until all nodes are determined
        while not mask.all():
            
            # random init
            e_noised = (torch.randn_like(e) > 0).float()
            
            # denoise steps
            for i in range(self.inference_steps):
                # consistency time
                t1, t2 = self.time_schedule(i)
                t1 = torch.tensor([t1], device=self.device).float()

                # add small random noise
                e_small_noise = 1.0 + self.cm_beta * torch.rand_like(e_noised)
                e_noised = e_noised * e_small_noise 

                # mask
                e_noised_mask = torch.where(mask, e-0.5, e_noised-0.5)
                
                # forward
                x_pred, e_pred = self.model.forward(
                    task=task, focus_on_node=False, focus_on_edge=True, 
                    nodes_feature=nodes_feature, x=x, edges_feature=graph, 
                    e=e_noised_mask, mask=mask, t=t1, edge_index=None
                )
                del x_pred
                
                # softmax
                e_pred_softmax = e_pred.softmax(1)

                # t2
                if t2 != 0:
                    e_pred_ber = torch.bernoulli(e_pred_softmax[:, 1, :].clamp(0, 1))
                    e_pred_ber_onehot: Tensor = F.one_hot(e_pred_ber.long(), num_classes=2)
                    Q_bar = torch.from_numpy(self.diffusion.Q_bar[t2]).float().to(self.device)
                    e_prob = torch.matmul(e_pred_ber_onehot.float(), Q_bar)
                    e_noised = torch.bernoulli(e_prob[..., 1].clamp(0, 1))

            # heatmap
            e_heatmap = e_pred_softmax[:, 1, :, :]
            
            # determine action
            continue_flag = not (current_determinate_steps < self.determinate_steps - 1)
            e, mask = self.env.update_mask_dense(
                heatmap=e_heatmap, e=e, raw_data_list=raw_data_list, 
                mask=mask, continue_flag=continue_flag,
                max_steps=max_steps_list[current_determinate_steps]
            )
            current_determinate_steps += 1
            if current_determinate_steps == self.determinate_steps:
                break
            
        # return loss and heatmap if mode is ``val``
        if self.env.mode == "val":
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(e_pred, ground_truth)
            return loss, e
        
        # return heatmap if mode is ``solve``
        if self.env.mode == "solve":
            return e

    def inference_node_sparse_process(
        self, task: str, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, e: Tensor, 
        edge_index: Tensor, graph_list: List[Tensor], mask: Tensor, ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list, raw_data_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # determinate_steps
        current_determinate_steps = 0
        max_steps = len(x) // self.determinate_steps
        
        # while loop, until all nodes are determined
        while not mask.all():
            # random init
            x_noised = (torch.randn_like(x) > 0).long().float()
            
            # inference denoise steps
            for i in range(self.inference_steps):
                # consistency time
                t1, t2 = self.time_schedule(i)
                t1 = torch.tensor([t1], device=self.device).float()
                
                # add small random noise
                x_small_noise = 1.0 + self.cm_beta * torch.rand_like(x_noised)
                x_noised = x_noised * x_small_noise
                
                # mask
                x_noised_mask = torch.where(mask, x-0.5, x_noised-0.5)

                # forward
                x_pred, e_pred = self.model.forward(
                    task=task, focus_on_node=True, focus_on_edge=False, 
                    nodes_feature=nodes_feature, x=x_noised_mask, 
                    edges_feature=edges_feature, e=e, mask=mask, 
                    t=t1, edge_index=edge_index
                )
                del e_pred
                
                # softmax
                x_pred_softmax = x_pred.softmax(-1)

                # t2
                if t2 != 0:
                    x_pred_ber = torch.bernoulli(x_pred_softmax[..., 1].clamp(0, 1))
                    x_pred_ber_onehot: Tensor = F.one_hot(x_pred_ber.long(), num_classes=2)
                    Q_bar = torch.from_numpy(self.diffusion.Q_bar[t2]).float().to(self.device)
                    x_prob = torch.matmul(x_pred_ber_onehot.float(), Q_bar)
                    x_noised = torch.bernoulli(x_prob[..., 1].clamp(0, 1))
        
            # heatmap
            x_heatmap = x_pred_softmax[:, 1]

            # determine action
            continue_flag = not (current_determinate_steps < self.determinate_steps - 1)
            x, mask = self.env.update_mask_sparse(
                heatmap=x_heatmap, x=x, e=e, graph_list=graph_list, mask=mask, 
                nodes_num_list=nodes_num_list, edges_num_list=edges_num_list,
                max_steps=max_steps, continue_flag=continue_flag, beam_size=self.beam_size
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
        
        
def get_tsp_max_steps_list(
    nodes_num: int, determinate_steps: int, inference_steps: int
) -> list:
    # if determinate_steps is 1, just return [-1]
    if determinate_steps == 1:
        return [-1]
    
    # assert determinate_steps is 3
    assert determinate_steps == 3
    
    # different nodes number
    if nodes_num <= 50:
        if inference_steps == 1:
            return [int(1.0 * nodes_num), int(1.7 * nodes_num), -1]
        else:
            return [int(0.8 * nodes_num), int(1.5 * nodes_num), -1]
    elif nodes_num <= 100:
        if inference_steps == 1:
            return [int(1.0 * nodes_num), int(1.7 * nodes_num), -1]
        else:
            return [int(0.7 * nodes_num), int(1.5 * nodes_num), -1]
    elif nodes_num <= 500:
        if inference_steps == 1:
            return [int(0.8 * nodes_num), int(1.9 * nodes_num), -1]
        else:
            return [int(0.7 * nodes_num), int(1.7 * nodes_num), -1]
    elif nodes_num <= 1500:
        if inference_steps == 1:
            return [int(0.6 * nodes_num), int(1.7 * nodes_num), -1]
        else:
            return [int(0.5 * nodes_num), int(1.5 * nodes_num), -1]
    else:
        return [int(0.4 * nodes_num), int(1.2 * nodes_num), -1]
    
    
def get_atsp_max_steps_list(
    nodes_num: int, determinate_steps: int, inference_steps: int
) -> list:
    # if determinate_steps is 1, just return [-1]
    if determinate_steps == 1:
        return [-1]
    
    # assert determinate_steps is 3
    assert determinate_steps == 3
    
    # different nodes number
    if nodes_num <= 50:
        if inference_steps == 1:
            return [int(0.7 * nodes_num), int(0.8 * nodes_num), -1]
        else:
            return [int(0.5 * nodes_num), int(0.7 * nodes_num), -1]
    elif nodes_num <= 100:
        if inference_steps == 1:
            return [int(0.7 * nodes_num), int(0.8 * nodes_num), -1]
        else:
            return [int(0.5 * nodes_num), int(0.7 * nodes_num), -1]
    elif nodes_num <= 200:
        if inference_steps == 1:
            return [int(0.6 * nodes_num), int(0.8 * nodes_num), -1]
        else:
            return [int(0.4 * nodes_num), int(0.6 * nodes_num), -1]
    else:
        if inference_steps == 1:
            return [int(0.5 * nodes_num), int(0.6 * nodes_num), -1]
        else:
            return [int(0.1 * nodes_num), int(0.2 * nodes_num), -1]
        

def get_cvrp_max_steps_list(
    nodes_num: int, determinate_steps: int, inference_steps: int
) -> list:
    # if determinate_steps is 1, just return [-1]
    if determinate_steps == 1:
        return [-1]
    
    # assert determinate_steps is 3
    assert determinate_steps == 3
    
    # assert inference_steps is 1
    assert inference_steps == 1
    
    # different nodes number
    if nodes_num <= 51:
        return [3, 5, -1]
    elif nodes_num <= 101:
        return [5, 9, -1]
    elif nodes_num <= 201:
        return [7, 12, -1]
    elif nodes_num <= 501:
        return [9, 16, -1]
    else:
        raise NotImplementedError()