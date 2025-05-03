import torch
from typing import Tuple
from torch import Tensor, nn
from tensordict import TensorDict
from symnco.model.embedder import get_init_embedder_by_task
from symnco.model.encoder.gat_layer import MultiHeadAttentionLayer


class GATEncoder(nn.Module):
    def __init__(
        self,
        task: str,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        force_flash_attn: bool = False
    ):
        super(GATEncoder, self).__init__()

        # embedder and out_layer
        self.task = task
        self.hidden_dim = hidden_dim
        self.embedder = get_init_embedder_by_task(task)(hidden_dim)
        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    hidden_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                    force_flash_attn=force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transfer to embedding space
        init_h = self.embedder(td)
        
        # Process embedding
        h = self.layers(init_h)

        # Return latent representation and initial embedding
        return h, init_h