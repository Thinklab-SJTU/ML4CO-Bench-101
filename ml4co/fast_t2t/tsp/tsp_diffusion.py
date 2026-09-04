import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor


class TSPDiffusion(object):
    def __init__(self, T: int = 1000):
        # Number of diffusion steps
        self.T = T

        # Set the initial and final beta values for the schedule
        b0 = 1e-4   # initial beta
        bT = 2e-2   # final beta

        # Linearly interpolate betas for each diffusion step
        self.beta = np.linspace(b0, bT, T)

        # Reshape beta for broadcasting in matrix operations
        beta = self.beta.reshape((-1, 1, 1))

        # Create identity matrix and all-ones matrix for the transition matrices
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        # Construct the transition matrices Qs for each time step
        self.Qs = (1 - beta) * eye + (beta / 2) * ones

        # Recursively compute the cumulative transition matrices 
        # Q_bar: Q_bar[t] = Q_1 * ... * Q_t
        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)

    def sample(self, x: Tensor, t1: Tensor, t2: Tensor):
        """
        Args:
            x: (E,); t1: (1,); t2: (1,)
        Returns:
            xt1: (E,); xt2: (E,)
        """
        # device
        device = x.device
        
        # Per-sample transition matrices
        Q_bar_t1 = torch.from_numpy(self.Q_bar[t1]).float().to(device)  # (2, 2)
        Q_bar_t2 = torch.from_numpy(self.Q_bar[t2]).float().to(device)  # (2, 2)
        mix_Q_bar_t_t2: Tensor = torch.linalg.inv(Q_bar_t2) @ Q_bar_t1  # (2, 2)
        
        # xt2
        x_onehot: Tensor = F.one_hot(x.long(), num_classes=2) # (E, 2)
        xt2 = torch.matmul(x_onehot.float(), Q_bar_t2) # (E, 2)
        xt2 = torch.bernoulli(xt2[..., 1].clamp(0, 1)) # (E, 2)
        xt2_onehot: Tensor = F.one_hot(xt2.long(), num_classes=2) # (E, 2)
        
        # xt1
        xt1 = torch.matmul(xt2_onehot.float(), mix_Q_bar_t_t2) # (E, 2)
        xt1 = torch.bernoulli(xt1[..., 1].clamp(0, 1)) # (E, 2)

        # reshape   
        return xt1, xt2

    def sample_single(self, x: Tensor, t: Tensor):
        """
        Args:
            x: (E,); t: (1,)
        Returns:
            xt: (E,)
        """
        # Sample a single diffusion step: xt ~ Q_bar[t] applied to x.
        device = x.device
        Q_bar_t = torch.from_numpy(self.Q_bar[t]).float().to(device)  # (2, 2)
        x_onehot: Tensor = F.one_hot(x.long(), num_classes=2)  # (E, 2)
        xt = torch.matmul(x_onehot.float(), Q_bar_t)  # (E, 2)
        xt = torch.bernoulli(xt[..., 1].clamp(0, 1))  # (E, 2)
        return xt