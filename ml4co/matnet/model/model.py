import torch
import numpy as np
from torch import nn, Tensor
from ml4co_kit import to_numpy
from dataclasses import dataclass
from matnet.env import MatNetATSPEnv
from matnet.model.encoder import MatNetATSPEncoder
from matnet.model.decoder import MatNetATSPDecoder


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor # shape: (batch, pomo)
    current_node: torch.Tensor = None # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None # shape: (batch, pomo, node)
    

class MatNetATSPModel(nn.Module):
    def __init__(
        self,
        env: MatNetATSPEnv,
        one_hot_seed_cnt: int,
        encoder_layers_num: int = 5,
        embedding_dim: int = 256, 
        ff_hidden_dim: int = 512, 
        qkv_dim: int = 16,
        head_num: int = 16, 
        ms_hidden_dim: int = 16, 
        mix1_init: float = 0.5**0.5, 
        mix2_init: float = 0.25,
        logit_clipping: int = 10,
        eval_type: str = "argmax",
        weight_path: str = None
    ):
        super().__init__()
        
        # info
        self.one_hot_seed_cnt = one_hot_seed_cnt
        self.embedding_dim = embedding_dim
        self.eval_type = eval_type
        
        # env
        self.env = env
        
        # encoder
        self.encoder = MatNetATSPEncoder(
            layers_num=encoder_layers_num,
            embedding_dim=embedding_dim,
            ff_hidden_dim=ff_hidden_dim,
            qkv_dim=qkv_dim,
            head_num=head_num,
            ms_hidden_dim=ms_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init 
        )
        
        # decoder
        self.decoder = MatNetATSPDecoder(
            embedding_dim=embedding_dim,
            qkv_dim=qkv_dim,
            head_num=head_num,
            logit_clipping=logit_clipping
        )

        # others
        self.encoded_row = None
        self.encoded_col = None
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)


    def pre_forward(self, dists: Tensor):
        batch_size, nodes_num, _ = dists.shape
        
        row_emb = torch.zeros(size=(batch_size, nodes_num, self.embedding_dim)).to(self.env.device)
        col_emb = torch.zeros(size=(batch_size, nodes_num, self.embedding_dim)).to(self.env.device)
        
        rand = torch.rand(batch_size, self.one_hot_seed_cnt).to(self.env.device)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :nodes_num]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, nodes_num)
        n_idx = torch.arange(nodes_num)[None, :].expand(batch_size, nodes_num)
        col_emb[b_idx, n_idx, rand_idx] = 1

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, dists)
        self.decoder.set_kv(self.encoded_col)

    def forward(self, state: StepState):

        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            selected = selected.to(self.env.device)
            prob = torch.ones(size=(batch_size, pomo_size)).to(self.env.device)

            # encoded_rows_mean = self.encoded_row.mean(dim=1, keepdim=True)
            # encoded_cols_mean = self.encoded_col.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            encoded_first_row = _get_encoding(self.encoded_row, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder.forward(
                encoded_current_row, ninf_mask=state.ninf_mask
            )
            # shape: (batch, pomo, job)

            if self.training or self.eval_type == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)

                    prob = all_job_probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break
            else:
                selected = all_job_probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

        
    def solve(self, dists: np.ndarray, beam_size: int) -> np.ndarray:
        # array -> tensor
        dists: Tensor = torch.from_numpy(dists).to(self.env.device)
        
        # preparation
        if dists.ndim == 2:
            dists = dists.unsqueeze(dim=0)
        batch_size, nodes_num, _ = dists.shape
        aug_factor = beam_size if beam_size > 1 else 1
        dists = dists.repeat(aug_factor, 1, 1)

        # pre forward
        self.env.load_problems_manual(dists)
        reset_state, _, _ = self.env.reset()
        self.pre_forward(reset_state)

        # pomo rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.forward(state)
            state, reward, done = self.env.step(selected)
        
        # reward
        reward = reward.reshape(aug_factor, batch_size, nodes_num)
        reward = reward.transpose(0, 1)
        reward = reward.reshape(batch_size, -1)
        
        # solution
        solution = torch.stack(self.env.tours).transpose(0, 1).transpose(1, 2)
        solution = solution.reshape(aug_factor, batch_size, nodes_num, nodes_num)
        solution = solution.transpose(0, 1)
        solution = solution.reshape(batch_size, -1, nodes_num)
        
        # best
        best_idx = torch.argmax(reward, dim=1)
        best_solution = solution[torch.arange(batch_size), best_idx]
  
        return to_numpy(best_solution)
        
        
def _get_encoding(
    encoded_nodes: Tensor, node_index_to_pick: Tensor
) -> Tensor:
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(
        batch_size, pomo_size, embedding_dim
    )
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes