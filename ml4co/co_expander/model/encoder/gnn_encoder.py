from torch import nn, Tensor
from typing import Sequence, Union
from co_expander.model.embedder import get_embedder_by_task
from co_expander.model.out_layer import get_out_layer_by_task
from co_expander.model.encoder.gnn_layer import GNNSparseBlock, GNNDenseBlock


class GNNEncoder(nn.Module):
    def __init__(
        self,
        task: str,
        sparse: bool,
        block_layers: Sequence[int],
        time_flag: bool = True,
        hidden_dim: int = 256, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False,
        mask_frozen: bool = False
    ):
        super(GNNEncoder, self).__init__()
        
        # info
        self.task = task
        self.sparse = sparse
        self.block_layers = block_layers
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        
        # embedder and out_layer
        self.embedder = get_embedder_by_task(task)(hidden_dim, sparse, time_flag)
        self.out_layer = get_out_layer_by_task(task)(hidden_dim, 2, sparse)
                
        # time embedder between blocks
        self.hidden_dim = hidden_dim
        self.node_time_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(len(block_layers))
        ])
        self.edge_time_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(len(block_layers))
        ])

        # asym
        if task in ["ATSP"]:
            asym = True
        else:
            asym = False
            
        # gnn blocks
        if sparse:
            # gnn sparse blocks
            self.blocks = nn.ModuleList([
                GNNSparseBlock(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    norm=norm,
                    learn_norm=learn_norm,
                    track_norm=track_norm,
                    asym=asym
                ) for num_layers in block_layers
            ])
        else:
            # gnn dense blocks
            self.blocks = nn.ModuleList([
                GNNDenseBlock(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    norm=norm,
                    learn_norm=learn_norm,
                    track_norm=track_norm,
                    asym=asym
                ) for num_layers in block_layers
            ])
        
        # mask layer
        nn.init.constant_(self.embedder.mask_embed.weight, 0)
        if mask_frozen:
            self.embedder.mask_embed.weight.requires_grad = False  
            
    def forward(
        self, task: str, focus_on_node: bool, focus_on_edge: bool, 
        nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        if task in ["ATSP"]:
            return self.asym_forward(
                task=task, focus_on_node=focus_on_node, focus_on_edge=focus_on_edge,
                nodes_feature=nodes_feature, x=x, edges_feature=edges_feature,
                e=e, mask=mask, t=t, edge_index=edge_index
            )
        else:
            return self.sym_forward(
                task=task, focus_on_node=focus_on_node, focus_on_edge=focus_on_edge,
                nodes_feature=nodes_feature, x=x, edges_feature=edges_feature,
                e=e, mask=mask, t=t, edge_index=edge_index
            )
        
    def asym_forward(
        self, task: str, focus_on_node: bool, focus_on_edge: bool, 
        nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # nodes number
        nodes_num = None if x is None else x.shape[0]
        
        # embedder
        e, t = self.embedder(nodes_feature, x, edges_feature, e, mask, t)
        
        # gnn blocks
        for gnn_block, node_time_layer, edge_time_layer in zip(
            self.blocks, self.node_time_layers, self.edge_time_layers
        ):
            gnn_block: Union[GNNDenseBlock, GNNSparseBlock]
            e = gnn_block.asym_forward(
                e=e, edges_feature=edges_feature, 
                edge_index=edge_index, nodes_num=nodes_num
            )
            if focus_on_node:
                raise NotImplementedError()
            if focus_on_edge and t is not None:
                e = e + edge_time_layer(t)

        # out layer
        x, e = self.out_layer(x, e)
        # return
        return x, e
    
    def sym_forward(
        self, task: str, focus_on_node: bool, focus_on_edge: bool, 
        nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # embedder
        x, e, t = self.embedder(nodes_feature, x, edges_feature, e, mask, t)

        # gnn blocks
        for gnn_block, node_time_layer, edge_time_layer in zip(
            self.blocks, self.node_time_layers, self.edge_time_layers
        ):
            gnn_block: Union[GNNDenseBlock, GNNSparseBlock]
            x, e = gnn_block.forward(x=x, e=e, edge_index=edge_index)
            if focus_on_node and t is not None:
                x = x + node_time_layer(t)
            if focus_on_edge and t is not None:
                e = e + edge_time_layer(t)

        # out layer
        x, e = self.out_layer(x, e)

        # return
        return x, e