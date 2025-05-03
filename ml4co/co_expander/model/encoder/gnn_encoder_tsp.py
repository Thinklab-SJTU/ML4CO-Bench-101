import torch
from typing import Sequence
from torch import nn, Tensor
from torch_sparse import SparseTensor
from co_expander.model.encoder.gnn_layer_tsp import GNNLayer
from co_expander.model.embedder.time import timestep_embedding
from co_expander.model.embedder.utils import PositionEmbeddingSine, ScalarEmbeddingSine3D


class TSPGNNEncoder(nn.Module):
    def __init__(
        self, 
        sparse: bool,
        num_layers: int = 12,
        hidden_dim: int = 256, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False,
        mask_frozen: bool = False
    ):
        super(TSPGNNEncoder, self).__init__()
        
        # info
        self.sparse = sparse
        self.hidden_dim = hidden_dim
        self.time_embed_dim = hidden_dim // 2
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        
        # embedder
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine3D(hidden_dim)
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.time_embed_dim,
                    hidden_dim,
                ),
            ) for _ in range(num_layers)
        ])

        # out layer
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
        )
        
        # gnn layers
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm)
            for _ in range(num_layers)
        ])
            
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
    
        # mask layer
        self.mask_embed = nn.Embedding(2, hidden_dim)
        nn.init.constant_(self.mask_embed.weight, 0)
        if mask_frozen:
            self.mask_embed.weight.requires_grad = False  
    
    def forward(
        self, task: str, focus_on_node: bool, focus_on_edge: bool, 
        nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # dense
        if self.sparse:
            # node embedder
            x: Tensor = self.pos_embed(nodes_feature.unsqueeze(0)) # (1, V, H)
            x = self.node_embed(x.squeeze(0)) # (V, H)
            
            # edge embedder
            e: Tensor = self.edge_pos_embed(e.expand(1, 1, -1)) # (1, E, H)
            e = self.edge_embed(e.squeeze()) # (E, H)
            mask = self.mask_embed(mask.long()) # (E, H)
            e = e + mask # (E, H)
            
            # time embedder
            t = self.time_embed(timestep_embedding(t, self.hidden_dim))
            
            # gnn layer
            edge_index = edge_index.long()
            adj_matrix = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(x.shape[0], x.shape[0]),
            )
            adj_matrix = adj_matrix.to(x.device)
            for layer, time_layer, out_layer in zip(
                self.layers, self.time_embed_layers, self.per_layer_out
            ):
                x_in, e_in = x, e
                x, e = layer(x_in, e_in, adj_matrix, edge_index=edge_index, sparse=True, mode="direct")
                e = e + time_layer(t)
                x = x_in + x
                e = e_in + out_layer(e)
            
            # out
            e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
            e = self.out(e)
            e = e.reshape(-1, edge_index.shape[1]).permute((1, 0))
        
        else:
            # embedder
            x = self.node_embed(self.pos_embed(nodes_feature)) # (B, V, H)
            e = self.edge_embed(self.edge_pos_embed(e)) # (B, V, V, H)
            mask = self.mask_embed(mask.long()) # (B, V, V, H)
            e = e + mask  # (B, V, H)
            t = self.time_embed(timestep_embedding(t, self.hidden_dim))
            graph = torch.ones_like(e).long()
            
            # gnn layer
            for layer, time_layer, out_layer in zip(
                self.layers, self.time_embed_layers, self.per_layer_out
            ):
                x_in, e_in = x, e
                x, e = layer(x, e, graph, mode="direct")
                e = e + time_layer(t)[:, None, None, :]
                x = x_in + x
                e = e_in + out_layer(e)

            # out layer
            e = self.out(e.permute((0, 3, 1, 2)))

        # return
        return x, e    


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type(x.dtype)
    

def normalization(channels: int):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
