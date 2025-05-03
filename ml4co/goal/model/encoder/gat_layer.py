import torch
from torch import Tensor, nn
from goal.model.encoder.attention import MultiHeadAttention


class ResidualNorm(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(
        self, state_before: Tensor, modified_after: Tensor
    ) -> Tensor:
        # residual connection
        state_before = state_before + self.alpha * modified_after
        # batch/layer norm
        return state_before
    

class GATLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        ff_dim: int = 512,
        num_heads: int = 8, 
        attention_clip_value: float = 10.0,
        ff_activation: str = "relu",
    ):
        super(GATLayer, self).__init__()

        # attention
        self.attn = MultiHeadAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            clip_value=attention_clip_value
        )
        
        # linear
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        
        # activation
        if ff_activation == "relu":
            self.activation = torch.nn.ReLU()
        elif ff_activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError()

        # norm
        self.res_norm1 = ResidualNorm()
        self.res_norm2 = ResidualNorm()

    def forward(
        self, x: Tensor, e: Tensor, mask: Tensor = None
    ) -> Tensor:
        # store
        x_rc = x
        
        # self attention
        x = self.attn.forward(x, x, e, mask)
        
        # residual + norm
        x = self.res_norm1(x_rc, x)
        x_rc = x
        
        # FF
        x = self.linear2(self.activation(self.linear1(x)))
        
        # residual + norm
        x = self.res_norm2(x_rc, x)
        
        return x