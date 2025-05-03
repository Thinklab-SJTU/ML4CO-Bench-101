from typing import Tuple
from torch import nn, Tensor
from matnet.model.layer import EncoderLayer


class MatNetATSPEncoder(nn.Module):
    def __init__(
        self, 
        layers_num: int,
        embedding_dim: int = 256, 
        ff_hidden_dim: int = 512, 
        qkv_dim: int = 16,
        head_num: int = 16, 
        ms_hidden_dim: int = 16, 
        mix1_init: float = 0.5**0.5, 
        mix2_init: float = 0.25
    ):
        super(MatNetATSPEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                embedding_dim=embedding_dim,
                ff_hidden_dim=ff_hidden_dim,
                qkv_dim=qkv_dim,
                head_num=head_num,
                ms_hidden_dim=ms_hidden_dim,
                mix1_init=mix1_init,
                mix2_init=mix2_init 
            ) for _ in range(layers_num)
        ])

    def forward(
        self, row_emb: Tensor, col_emb: Tensor, cost_mat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)
        return row_emb, col_emb
    
