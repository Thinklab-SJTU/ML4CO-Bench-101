import torch
from torch import nn, Tensor
from goal.model.adapter import (
    get_node_adapter_by_task, 
    get_edge_adapter_by_task, 
    get_output_adapter_by_task
)
from goal.model.encoder.gat_layer import GATLayer
from goal.model.adapter import (
    GOALNodeAdapter, GOALEdgeAdapter, GOALOutputAdapter
)


class GATEncoder(nn.Module):
    def __init__(
        self, 
        task: str,
        node_idx_dim: int = 1, 
        hidden_dim: int = 128, 
        ff_dim: int = 512,
        node_low_dim: int = 8,
        edge_low_dim: int = 4,
        num_layers: int = 9, 
        num_heads: int = 8, 
        attention_clip_value: float = 10.0,
        ff_activation: str = "relu",
        edge_adapter_activation: str = "relu"
    ):
        super().__init__()

        # init
        self.task = task
        self.node_idx_dim = node_idx_dim
        self.node_idx_projection = nn.Linear(node_idx_dim, hidden_dim)
        self.num_layers = num_layers

        # adapters
        self.node_adapter: GOALNodeAdapter = get_node_adapter_by_task(task)(
            hidden_dim, node_low_dim
        )
        self.edge_adapter: GOALEdgeAdapter = get_edge_adapter_by_task(task)(
            hidden_dim, edge_low_dim, edge_adapter_activation
        )
        self.output_adapter: GOALOutputAdapter = get_output_adapter_by_task(task)(hidden_dim)

        # gat layers
        self.layers = nn.modules.ModuleList([
            GATLayer(hidden_dim, ff_dim, num_heads, attention_clip_value, ff_activation)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, e: Tensor, mask: Tensor = None):
        # data info
        feature = e if e is not None else x
        batch_size, seq_len, device = feature.shape[0], feature.shape[1], feature.device

        # random embedding
        random_embedding = self.node_idx_projection(
            torch.rand((batch_size, seq_len, self.node_idx_dim), device=device)
        )
        # input adapters
        x = self.node_adapter.forward(
            nodes_feature=x, random_embedding=random_embedding
        )
        e = self.edge_adapter.forward(edges_feature=e)

        # backbone
        for layer in self.layers:
            x = layer(x, e, mask)

        # output adapter
        scores = self.output_adapter.forward(state=x)

        # masking infeasible actions
        scores = self.mask_infeasible_actions(scores, mask)
        
        return scores.reshape(scores.shape[0], -1)
    
    def mask_infeasible_actions(self, scores: Tensor, mask: Tensor) -> Tensor:
        if self.task in ["TSP", "ATSP", "CVRP"]:
            scores[:, 0] = scores[:, -1] = -torch.inf
        elif self.task in ["MIS", "MCl"]:
            if mask is None:
                return scores
            scores[mask[..., 0]] = -torch.inf
        elif self.task in ["MVC"]:
            if mask is None:
                return scores
            scores[mask[..., 0]] = torch.inf
        else:
            raise NotImplementedError()
        return scores