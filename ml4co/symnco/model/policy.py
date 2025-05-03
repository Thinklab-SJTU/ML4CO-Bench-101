from torch import nn, Tensor
from tensordict.tensordict import TensorDict
from symnco.utils.mlp import MLP
from symnco.env.base import SymNCOEnv
from symnco.model.decoder.decoder import SymNCODecoder
from symnco.model.encoder.gat_encoder import GATEncoder


def get_log_likelihood(
    log_p: Tensor, actions: Tensor, mask: Tensor, return_sum: bool = True
) -> Tensor:
    """Get log likelihood of selected actions"""

    log_p = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[~mask] = 0

    assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return log_p.sum(1)  # [batch]
    else:
        return log_p  # [batch, decode_len]
    

class SymNCOPolicy(nn.Module):
    def __init__(
        self,
        encoder: GATEncoder,
        decoder: SymNCODecoder,
        decode_type: str = "greedy"
    ):
        super(SymNCOPolicy, self).__init__()
        self.encoder: GATEncoder = encoder
        self.decoder: SymNCODecoder = decoder
        self.decode_type = decode_type
        self.hidden_dim = self.encoder.hidden_dim
        self.projection_head = MLP(
            in_features=self.hidden_dim, 
            out_features=self.hidden_dim, 
            depth=1, 
            num_cells=self.hidden_dim, 
            activation_class=nn.ReLU
        )

    def setup(self, stage="fit"):
        pass
        
    def forward(
        self,
        td: TensorDict,
        env: SymNCOEnv,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = True,
        **decoder_kwargs,
    ) -> dict:
        super().forward.__doc__  # trick to get docs from parent class

        # Encoder: get embeddings from initial state
        embeddings, init_embeds = self.encoder.forward(td)
        log_p, actions, td_out = self.decoder.forward(
            td, embeddings, env, self.decode_type, **decoder_kwargs
        )
        
        # Log likelihood is calculated within the model
        log_p: Tensor
        td_out: TensorDict
        log_likelihood = get_log_likelihood(log_p, actions, td_out.get("mask", None))
        out = {
            "reward": td_out["reward"],
            "log_likelihood": log_likelihood,
        }
        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(log_p.exp() * log_p).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        if return_init_embeds:
            out["init_embeds"] = init_embeds

        # Project initial embeddings
        out["proj_embeddings"] = self.projection_head(out["init_embeds"])

        return out
