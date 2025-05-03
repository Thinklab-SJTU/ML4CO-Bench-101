import torch
import torch.nn.functional as F
from typing import Tuple
from torch import nn, Tensor


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        added = input1 + input2 # (batch, problem, embedding)
        transposed = added.transpose(1, 2) # (batch, embedding, problem)
        normalized: Tensor = self.norm(transposed) # (batch, embedding, problem)
        back_trans = normalized.transpose(1, 2) # (batch, problem, embedding)
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int = 256, ff_hidden_dim: int = 512):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: Tensor) -> Tensor:
        return self.W2(F.relu(self.W1(input1)))


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        qkv_dim: int = 16,
        head_num: int = 16, 
        ms_hidden_dim: int = 16, 
        mix1_init: float = 0.5**0.5, 
        mix2_init: float = 0.25
    ):
        super().__init__()
        # info
        self.qkv_dim = qkv_dim
        self.sqrt_qkv_dim = qkv_dim ** 0.5
        self.head_num = head_num
        self.ms_hidden_dim = ms_hidden_dim
        
        # mix1
        mix1_weight = torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)

        # mix2
        mix2_weight = torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, cost_mat: Tensor
    ) -> Tensor:
        """
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        """
        # info
        batch_size, _, row_cnt, _ = q.shape
        col_cnt = k.shape[2]
        
        # process
        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / self.sqrt_qkv_dim
        cost_mat_score = cost_mat[:, None, :, :].expand(
            batch_size, self.head_num, row_cnt, col_cnt
        )
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        two_scores_transposed = two_scores.transpose(1, 2)
         
        # ms1
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        ms1_activated = F.relu(ms1)

        # ms2
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        
        # mix
        mixed_scores = ms2.transpose(1, 2)
        mixed_scores = mixed_scores.squeeze(4)
        weights = nn.Softmax(dim=3)(mixed_scores)
        
        # out
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        out_concat = out_transposed.reshape(batch_size, row_cnt, self.head_num * self.qkv_dim)

        return out_concat


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256, 
        ff_hidden_dim: int = 512, 
        qkv_dim: int = 16,
        head_num: int = 16, 
        ms_hidden_dim: int = 16, 
        mix1_init: float = 0.5**0.5, 
        mix2_init: float = 0.25
    ):
        super().__init__()
        self.row_encoding_block = EncodingBlock(
            embedding_dim=embedding_dim,
            ff_hidden_dim=ff_hidden_dim,
            qkv_dim=qkv_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init
        )
        self.col_encoding_block = EncodingBlock(
            embedding_dim=embedding_dim,
            ff_hidden_dim=ff_hidden_dim,
            qkv_dim=qkv_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init
        )

    def forward(
        self, row_emb: Tensor, col_emb: Tensor, cost_mat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        """
        row_emb_out = self.row_encoding_block(
            row_emb, col_emb, cost_mat
        )
        col_emb_out = self.col_encoding_block(
            col_emb, row_emb, cost_mat.transpose(1, 2)
        )
        return row_emb_out, col_emb_out
    

class EncodingBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim: int = 256, 
        ff_hidden_dim: int = 512, 
        qkv_dim: int = 16,
        head_num: int = 16, 
        ms_hidden_dim: int = 16, 
        mix1_init: float = 0.5**0.5, 
        mix2_init: float = 0.25
    ):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(
            qkv_dim=qkv_dim, head_num=head_num, ms_hidden_dim=ms_hidden_dim,
            mix1_init=mix1_init, mix2_init=mix2_init
        )
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(embedding_dim=embedding_dim)
        self.feed_forward = FeedForward(embedding_dim=embedding_dim, ff_hidden_dim=ff_hidden_dim)
        self.add_n_normalization_2 = AddAndInstanceNormalization(embedding_dim=embedding_dim)

    def forward(
        self, row_emb: Tensor, col_emb: Tensor, cost_mat: Tensor
    ) -> Tensor:
        # qkv
        q = reshape_by_heads(self.Wq(row_emb), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(col_emb), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=self.head_num)
        
        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        multi_head_out = self.multi_head_combine(out_concat)
        
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        
        
def reshape_by_heads(qkv: Tensor, head_num: int) -> Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed