import torch
import torch.nn.functional as F
import torch.nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import GINConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_min, scatter_max
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, dropout_adj
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from .utils import get_mask


class GNNEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 4, 
        hidden_dim_1: int = 64, 
        hidden_dim_2: int = 1, 
        deltas: float = 1.,
        momemtum: float = 0.1,
        num_heads: int = 8, 
        devide: str = "cuda"
    ):
        super(GNNEncoder, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.momentum = momemtum
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.heads = num_heads
        self.concat = True
        self.device = devide
        self.receptive_field = num_layers + 1
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.bns.append(BN(self.heads * self.hidden_dim_1, momentum=self.momentum))
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.heads * self.hidden_dim_1, self.heads * self.hidden_dim_1),
                        ReLU(),
                        Linear(self.heads * self.hidden_dim_1, self.heads * self.hidden_dim_1),
                        ReLU(),
                        BN(self.heads * self.hidden_dim_1, momentum=self.momentum),
                    ),
                    train_eps=True
                )
            )
        self.bn1 = BN(self.heads * self.hidden_dim_1)       
        self.conv1 = GINConv(
            Sequential(
                Linear(
                    self.hidden_dim_2, self.heads * self.hidden_dim_1),
                    ReLU(),
                    Linear(self.heads * self.hidden_dim_1, self.heads * self.hidden_dim_1),
                    ReLU(),
                    BN(self.heads * self.hidden_dim_1, momentum=self.momentum),
            ),
            train_eps=True
        )

        if self.concat:
            self.lin1 = Linear(self.heads * self.hidden_dim_1, self.hidden_dim_1)
        else:
            self.lin1 = Linear(self.hidden_dim_1, self.hidden_dim_1)
        self.lin2 = Linear(self.hidden_dim_1, 1)
        self.gnorm = GraphSizeNorm()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters() 
        for bn in self.bns:
            bn.reset_parameters()
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch, edge_dropout=None):
        N_size = x.shape[0]

        if edge_dropout is not None:
            edge_index = dropout_adj(
                edge_index,
                edge_attr=(torch.ones(edge_index.shape[1], device=self.device)).long(), 
                p=edge_dropout, 
                force_undirected=True
            )[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]  

        x = x.unsqueeze(-1)
        mask = get_mask(x,edge_index,1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = x * mask
        x = self.gnorm(x)
        x = self.bn1(x)
            
        for conv, bn in zip(self.convs, self.bns):
            if x.dim() > 1:
                x = x + F.leaky_relu(conv(x, edge_index))
                mask = get_mask(mask,edge_index,1).to(x.dtype)
                x = x * mask
                x = self.gnorm(x)
                x = bn(x)

        x = F.leaky_relu(self.lin1(x)) 
        x = x * mask

        x = F.leaky_relu(self.lin2(x)) 
        x = x * mask

        # calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size=N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size=N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

        # min-max normalize
        x = (x - batch_min) / (batch_max + 1e-6 - batch_min)
        probs = x

        return probs

