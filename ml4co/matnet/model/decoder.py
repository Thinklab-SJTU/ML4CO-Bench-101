import torch
import torch.nn.functional as F
from torch import nn, Tensor
from matnet.model.layer import reshape_by_heads


class MatNetATSPDecoder(nn.Module):
    def __init__(
        self, 
        embedding_dim: int = 256, 
        qkv_dim: int = 16,
        head_num: int = 16,
        logit_clipping: int = 10
    ):
        super(MatNetATSPDecoder, self).__init__()
        
        # info
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.sqrt_qkv_dim = qkv_dim ** 0.5
        self.embedding_dim = embedding_dim
        self.sqrt_embedding_dim = embedding_dim ** 0.5
        self.logit_clipping = logit_clipping
        
        # params
        self.Wq_0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # qkv
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_jobs: Tensor):
        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=self.head_num)
        self.single_head_key = encoded_jobs.transpose(1, 2)

    def set_q1(self, encoded_q1: Tensor):
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=self.head_num)

    def forward(self, encoded_q0: Tensor, ninf_mask: Tensor):
        # multi-head attention
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=self.head_num)
        q = self.q1 + q0
        out_concat = self._multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # single-head attention, for probability calculation
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / self.sqrt_embedding_dim
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs

    def _multi_head_attention(
        self, q: Tensor, k: Tensor, v: Tensor, 
        rank2_ninf_mask: Tensor = None, rank3_ninf_mask: Tensor = None
    ) -> Tensor:
        """
        # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or pomo
        # k,v shape: (batch, head_num, node, key_dim)
        # rank2_ninf_mask.shape: (batch, node)
        # rank3_ninf_mask.shape: (batch, group, node)
        """
        # info
        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)
        
        # score
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / self.sqrt_qkv_dim
        if rank2_ninf_mask is not None:
            right_term_1 = rank2_ninf_mask[:, None, None, :].expand(batch_s, self.head_num, n, node_cnt)
            score_scaled = score_scaled + right_term_1
        if rank3_ninf_mask is not None:
            right_term_2 = rank3_ninf_mask[:, None, :, :].expand(batch_s, self.head_num, n, node_cnt)
            score_scaled = score_scaled + right_term_2

        # out
        weights = nn.Softmax(dim=3)(score_scaled)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        out_concat = out_transposed.reshape(batch_s, n, self.head_num * self.qkv_dim)

        return out_concat