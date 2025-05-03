import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .sag_layer import SCTConv


class SAGEncoder(nn.Module):
    def __init__(
        self, 
        output_channels: int,
        num_layers: int = 3, 
        input_dim: int = 2, 
        hidden_dim: int = 64, 
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SCTConv(hidden_dim))

        self.mlp1 = nn.Linear(hidden_dim * (1 + num_layers), hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_channels)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.m = nn.Softmax(dim=1)

    def forward(self, x: Tensor, graph: Tensor) -> Tensor:
        x = self.in_proj(x)
        hidden_states = x
        for layer in self.convs:
            x = layer(x, graph)
            hidden_states = torch.cat([hidden_states, x], dim=-1)
        x = hidden_states
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        x = self.mlp2(x)
        x = self.m(x)
        return x