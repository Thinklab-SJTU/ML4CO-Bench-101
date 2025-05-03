import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from typing import Tuple, Union
from tensordict import TensorDict
from dataclasses import dataclass
from pytorch_lightning.utilities import rank_zero_info
from symnco.env.base import SymNCOEnv
from symnco.model.decoder.logit_attn import LogitAttention
from symnco.utils.ops import batchify, get_num_starts, select_start_nodes, unbatchify
from symnco.model.embedder import get_context_embedder_by_task, SymNCODynamicEmbedder


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


def decode_probs(probs: torch.Tensor, mask: torch.Tensor, decode_type: str="sampling"):
    """Decode probabilities to select actions with mask"""

    assert (probs == probs).all(), "Probs should not contain any nans"

    if "greedy" in decode_type:
        selected: Tensor = probs.max(1)[1]
        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "Decode greedy: infeasible action has maximum probability"

    elif "sampling" in decode_type:
        selected = torch.multinomial(probs, 1).squeeze(1)

        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            print("Sampled bad values, resampling!")
            selected = probs.multinomial(1).squeeze(1)

    else:
        assert False, "Unknown decode type: {}".format(decode_type)
    return selected


class SymNCODecoder(nn.Module):
    def __init__(
        self,
        task: str,
        hidden_dim: int = 128,
        use_graph_context: bool = True,
        select_start_nodes_fn: callable = select_start_nodes,
        linear_bias: bool = False,
        num_heads: int = 8,
        **logit_attn_kwargs,
    ):
        super(SymNCODecoder, self).__init__()

        # embeddings
        self.task = task
        self.hidden_dim = hidden_dim
        self.context_embedding = get_context_embedder_by_task(task)(hidden_dim)
        self.dynamic_embedding = SymNCODynamicEmbedder()
        self.use_graph_context = use_graph_context

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * hidden_dim
        self.project_node_embeddings = nn.Linear(hidden_dim, 3 * hidden_dim, bias=linear_bias)
        self.project_fixed_context = nn.Linear(hidden_dim, hidden_dim, bias=linear_bias)

        # MHA
        self.logit_attention = LogitAttention(hidden_dim, num_heads, **logit_attn_kwargs)

        # Others
        self.select_start_nodes_fn = select_start_nodes_fn

    def forward(
        self,
        td: TensorDict,
        embeddings: Tensor,
        env: SymNCOEnv,
        decode_type: str = "sampling",
        num_starts: int = None,
        softmax_temp: float = None,
        calc_reward: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, TensorDict, Tensor]]:
        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        if "multistart" in decode_type:
            if num_starts is None:
                if self.task == "CVRP":
                    num_starts = get_num_starts(td) - 1
                elif self.task == "TSP":
                    num_starts = get_num_starts(td)
        else:
            if num_starts is not None:
                if num_starts > 1:
                    rank_zero_info(
                        f"num_starts={num_starts} is ignored for decode_type={decode_type}"
                    )
            num_starts = 0

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings, num_starts=num_starts)

        # Collect outputs
        outputs = []
        actions = []
            
        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if num_starts > 1 or "multistart" in decode_type:
            action = self.select_start_nodes_fn(td, nodes_num=num_starts)

            # Expand td to batch_size * num_starts
            td = batchify(td, num_starts)

            td.set("action", action)
            td = env.step(td)["next"]
            log_p = torch.zeros_like(
                td["action_mask"], device=td.device
            )  # first log_p is 0, so p = log_p.exp() = 1

            outputs.append(log_p)
            actions.append(action)

        # Main decoding: loop until all sequences are done
        while not td["done"].all():
            log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts)
            
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            log_p: Tensor
            action = decode_probs(log_p.exp(), mask, decode_type=decode_type)
            td.set("action", action)
            td = env.step(td)["next"]

            # Collect output of step
            outputs.append(log_p)
            actions.append(action)

        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        if calc_reward:
            reward = env.get_reward(td, actions)
            if td.batch_size == torch.Size([1]):
                reward = torch.tensor([reward])
            td.set("reward", reward)

        return outputs, actions, td

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0):
        """Compute the cached embeddings for the attention

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of multi-starts to use. If 0, no multi-start decoding is used
        """

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = unbatchify(
                batchify(self.project_fixed_context(embeddings.mean(1)), num_starts),
                num_starts,
            )
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(
        self,
        cached: PrecomputedCache,
        td: TensorDict,
        softmax_temp: float = None,
        num_starts: int = 0,
    ):
        """Compute the log probabilities of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            softmax_temp: Temperature for the softmax
            num_starts: Number of starts for the multi-start decoding
        """
        # Unbatchify to [batch_size, num_starts, ...]. Has no effect if num_starts = 0
        td_unbatch = unbatchify(td, num_starts)
        step_context = self.context_embedding(cached.node_embeddings, td_unbatch)
        glimpse_q: Tensor = step_context + cached.graph_context
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        
        # Compute keys and values for the nodes
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td_unbatch)
        glimpse_k = cached.glimpse_key + glimpse_key_dynamic
        glimpse_v = cached.glimpse_val + glimpse_val_dynamic
        logit_k = cached.logit_key + logit_key_dynamic
        
        # Get the mask
        mask = ~td_unbatch["action_mask"]
        
        # Compute logits
        log_p = self.logit_attention(
            glimpse_q, glimpse_k, glimpse_v, logit_k, mask, softmax_temp
        )
        
        # Now we need to reshape the logits and log_p to [batch_size*num_starts, num_nodes]
        # Note that rearranging order is important here
        log_p = rearrange(log_p, "b s l -> (s b) l") if num_starts > 1 else log_p
        mask = rearrange(mask, "b s l -> (s b) l") if num_starts > 1 else mask
        return log_p, mask

    def __repr__(self):
        return f"{self.__class__.__name__}"