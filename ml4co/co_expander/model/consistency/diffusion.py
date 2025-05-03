import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor


class CategoricalDiffusion(object):
    def __init__(self, T: int, schedule: str):
        # Diffusion steps
        self.T = T

        # Noise schedule
        if schedule == "linear":
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == "cosine":
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
                0
            )  # Generate an extra alpha for bT
            self.beta = np.clip(
                1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999
            )

        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def __cos_noise(self, t: int) -> np.ndarray:
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def consistency_sample_sparse(self, x: Tensor, t1: Tensor, t2: Tensor):
        """
        Args:
            x: (N,); t: (1,); t2: (1,)
        Returns:
            xt1: (N,); xt2: (N,);
        """
        # device
        device = x.device
        
        # Q_bar
        Q_bar_t1 = torch.from_numpy(self.Q_bar[t1]).float().to(device) # (2, 2)
        Q_bar_t2 = torch.from_numpy(self.Q_bar[t2]).float().to(device) # (2, 2)
        mix_Q_bar_t_t2: Tensor = (torch.linalg.inv(Q_bar_t2) @ Q_bar_t1) # (2, 2)
        
        # xt2
        x_onehot: Tensor = F.one_hot(x.long(), num_classes=2) # (B, N, 2)  
        xt2 = torch.matmul(x_onehot.float(), Q_bar_t2)
        xt2 = torch.bernoulli(xt2[..., 1].clamp(0, 1))
        xt2_onehot: Tensor = F.one_hot(xt2.long(), num_classes=2)
        
        # xt1
        xt1 = torch.matmul(xt2_onehot.float(), mix_Q_bar_t_t2)
        xt1 = torch.bernoulli(xt1[..., 1].clamp(0, 1))

        return xt1, xt2
    
    def consistency_sample_dense(self, x: Tensor, t1: Tensor, t2: Tensor):
        """
        Args:
            x: (N,); t: (1,); t2: (1,)
        Returns:
            xt1: (N,); xt2: (N,);
        """
        # device
        device = x.device
        
        # Q_bar
        Q_bar_t1 = torch.from_numpy(self.Q_bar[t1]).float().to(device) # (2, 2)
        Q_bar_t2 = torch.from_numpy(self.Q_bar[t2]).float().to(device) # (2, 2)
        mix_Q_bar_t_t2: Tensor = (torch.linalg.inv(Q_bar_t2) @ Q_bar_t1) # (2, 2)
        
        # xt2
        x_onehot: Tensor = F.one_hot(x.long(), num_classes=2) # (B, N, 2)  
        xt2 = torch.matmul(x_onehot.float(), Q_bar_t2)
        xt2 = torch.bernoulli(xt2[..., 1].clamp(0, 1))
        xt2_onehot: Tensor = F.one_hot(xt2.long(), num_classes=2)
        
        # xt1
        xt1 = torch.matmul(xt2_onehot.float(), mix_Q_bar_t_t2)
        xt1 = torch.bernoulli(xt1[..., 1].clamp(0, 1))

        return xt1, xt2