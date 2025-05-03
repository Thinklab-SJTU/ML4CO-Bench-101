import torch
from typing import Any
from torch import Tensor
from einops import rearrange
from typing import Union, Tuple
from ml4co_kit import BaseModel, to_numpy
from tensordict.tensordict import TensorDict
from torch.nn.functional import cosine_similarity
from symnco.env.base import SymNCOEnv
from symnco.model.policy import SymNCOPolicy
from symnco.utils.transforms import StateAugmentation
from symnco.model.decoder.decoder import SymNCODecoder
from symnco.model.encoder.gat_encoder import GATEncoder
from symnco.utils.ops import get_num_starts, unbatchify, gather_by_index


class SymNCOModel(BaseModel):
    def __init__(
        self,
        env: SymNCOEnv,
        encoder: GATEncoder,
        decoder: SymNCODecoder,
        decode_type: str = "greedy",
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None,
        num_augment: int = 4,
        alpha: float = 0.2,
        beta: float = 1,
        num_starts: int = 0
    ):
        super(SymNCOModel, self).__init__(
            env=env,
            model=SymNCOPolicy(
                encoder=encoder,
                decoder=decoder,
                decode_type=decode_type
            ),
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: SymNCOEnv
        self.model: SymNCOPolicy
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

        # special
        self.num_augment = num_augment
        self.alpha = alpha
        self.beta = beta
        self.num_starts = num_starts
        self.augment = StateAugmentation(num_augment=self.num_augment)
        self.train_metrics = ["loss", "reward"]
        self.val_metrics = ["reward"]
        self.test_metrics = ["reward"]
        
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        task = self.env.task
        
        # get real data
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            td_data = self.env.generate_train_data(batch_size)
            
            # deal with different task
            if task in ["TSP", "CVRP"]:
                out = self.train_edge_process(td_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # get val data
            td_data = self.env.generate_val_data(batch_idx)

            # deal with different task
            if task in ["TSP", "CVRP"]:
                out = self.inference_edge_process(td_data)
            else:
                raise NotImplementedError()
            
        else:
            raise NotImplementedError()
     
        # log
        metrics = self.get_metrics(out, phase)
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return {"loss": out.get("loss", None), **metrics}
    
    def get_metrics(self, metric_dict: dict, phase: str):
        """Log metrics to logger and progress bar"""
        need_to_log = getattr(self, f"{phase}_metrics")
        metrics = {f"{phase}/{k}": v.mean() for k, v in metric_dict.items() if k in need_to_log}
        return metrics
        
    def train_edge_process(self, td: TensorDict) -> dict:
        td = self.env.reset(td)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td) if n_start is None else n_start

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out: dict = self.model.forward(td=td, env=self.env, num_starts=n_start)
        
        # Unbatchify reward to [batch_size, n_start, n_aug].
        reward = unbatchify(out["reward"], (n_start, n_aug))
        
        # [batch_size, n_start, n_aug]
        ll = unbatchify(out["log_likelihood"], (n_start, n_aug))

        # Calculate losses: problem symmetricity, solution symmetricity, invariance
        loss_ps = problem_symmetricity_loss(reward, ll) if n_start > 1 else 0
        loss_ss = solution_symmetricity_loss(reward, ll) if n_aug > 1 else 0
        loss_inv = invariance_loss(out["proj_embeddings"], n_aug) if n_aug > 1 else 0
        loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv
        out.update(
            {
                "loss": loss,
                "loss_ss": loss_ss,
                "loss_ps": loss_ps,
                "loss_inv": loss_inv,
            }
        )
        return out
        
    def inference_edge_process(self, td: TensorDict) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        td = self.env.reset(td)
        n_aug = self.num_augment
        if "multistart" in self.model.decode_type:
            n_start = td['locs'].shape[1] # nodes number
        else:
            n_start = 0
        n_start = get_num_starts(td) if n_start is None else n_start

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out: dict = self.model.forward(
            td=td, env=self.env, num_starts=n_start, return_actions=True
        )
        
        # Unbatchify reward to [batch_size, n_start, n_aug].
        reward = unbatchify(out["reward"], (n_start, n_aug))
        actions = unbatchify(out["actions"], (n_start, n_aug))
        
        if n_start > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=1)
            out.update({"max_reward": max_reward})

            # Reshape batch to [batch, n_start, n_aug]
            if out.get("actions", None) is not None:
                out.update(
                    {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if n_aug > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if n_start > 1 else reward
            max_aug_reward, max_idxs = reward_.max(dim=1)
            out.update({"max_aug_reward": max_aug_reward})
            if out.get("best_multistart_actions", None) is not None:
                out.update(
                    {
                        "best_aug_actions": gather_by_index(
                            out["best_multistart_actions"], max_idxs
                        )
                    }
                )
            
        if self.env.mode == "val":
            return out
        if self.env.mode == "solve":
            bs = actions.shape[0]
            nodes_num = actions.shape[-1]
            batch_actions = actions.reshape(bs, -1, nodes_num)
            batch_rewards = reward.reshape(bs, -1)
            best_idx = batch_rewards.max(1)[1]
            actions = batch_actions[torch.arange(bs), best_idx]
            return to_numpy(actions)
        else:
            raise ValueError()
        

def problem_symmetricity_loss(
    reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """REINFORCE loss for problem symmetricity
    Baseline is the average reward for all augmented problems
    Corresponds to `L_ps` in the SymNCO paper
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def solution_symmetricity_loss(
    reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """REINFORCE loss for solution symmetricity
    Baseline is the average reward for all start nodes
    Corresponds to `L_ss` in the SymNCO paper
    """
    num_starts = reward.shape[dim]
    if num_starts < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def invariance_loss(
    proj_embed: torch.Tensor, num_augment: int
) -> torch.Tensor:
    """Loss for invariant representation on projected nodes
    Corresponds to `L_inv` in the SymNCO paper
    """
    pe = rearrange(proj_embed, "(b a) ... -> b a ...", a=num_augment)
    similarity = sum(
        [cosine_similarity(pe[:, 0], pe[:, i], dim=-1) for i in range(1, num_augment)]
    )
    return similarity.mean()