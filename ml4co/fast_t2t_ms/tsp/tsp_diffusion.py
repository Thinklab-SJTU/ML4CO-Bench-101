import numpy as np
import mindspore as ms
from mindspore import Tensor, ops
from ml4co.ms_utils import ensure_ms_device, maybe_move_tensor, normalize_ms_device


class TSPDiffusion(object):
    def __init__(self, T: int = 1000):
        # Number of diffusion steps
        self.T = T

        # Set the initial and final beta values for the schedule
        b0 = 1e-4  # initial beta
        bT = 2e-2  # final beta

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
        self.Q_bar = np.stack(Q_bar, axis=0).astype(np.float32)

        # Device-resident copy; filled by ``to_device``.
        self.Q_bar_ms = None
        self._device = None

    def to_device(self, device: str):
        """Cache ``Q_bar`` as a MindSpore Tensor on ``device``."""
        device = normalize_ms_device(device)
        ensure_ms_device(device)
        self.Q_bar_ms = maybe_move_tensor(
            ms.Tensor(self.Q_bar, ms.float32), device, strict=True
        )
        self._device = device
        return self

    def q_bar_at(self, t) -> Tensor:
        """Return ``Q_bar[t]`` on the active device (no host re-alloc each call)."""
        t_i = int(t.asnumpy().reshape(-1)[0]) if isinstance(t, Tensor) else int(t)
        if self.Q_bar_ms is not None:
            return self.Q_bar_ms[t_i]
        # Fallback: allocate on current process device.
        return ms.Tensor(self.Q_bar[t_i], ms.float32)

    def sample(self, x: Tensor, t1, t2):
        """
        Args:
            x: (E,); t1 / t2: int or Tensor scalar
        Returns:
            xt1: (E,); xt2: (E,)
        """
        # Prefer Python ints to avoid Ascend D2H sync every step.
        t1_i = int(t1.asnumpy().reshape(-1)[0]) if isinstance(t1, Tensor) else int(t1)
        t2_i = int(t2.asnumpy().reshape(-1)[0]) if isinstance(t2, Tensor) else int(t2)

        Q_bar_t1 = self.q_bar_at(t1_i)  # (2, 2)
        Q_bar_t2 = self.q_bar_at(t2_i)  # (2, 2)
        mix_Q_bar = ops.matmul(ops.inv(Q_bar_t2), Q_bar_t1)  # (2, 2)

        # xt2
        x_onehot = ops.one_hot(
            x.astype(ms.int32),
            2,
            ms.Tensor(1.0, ms.float32),
            ms.Tensor(0.0, ms.float32),
        )  # (E, 2)
        xt2 = ops.matmul(x_onehot.astype(ms.float32), Q_bar_t2)  # (E, 2)
        xt2 = ops.bernoulli(ops.clip_by_value(xt2[..., 1], 0.0, 1.0))
        xt2_onehot = ops.one_hot(
            xt2.astype(ms.int32),
            2,
            ms.Tensor(1.0, ms.float32),
            ms.Tensor(0.0, ms.float32),
        )  # (E, 2)

        # xt1
        xt1 = ops.matmul(xt2_onehot.astype(ms.float32), mix_Q_bar)  # (E, 2)
        xt1 = ops.bernoulli(ops.clip_by_value(xt1[..., 1], 0.0, 1.0))
        return xt1, xt2

    def sample_single(self, x: Tensor, t):
        """
        Args:
            x: (E,); t: int or Tensor scalar
        Returns:
            xt: (E,)
        """
        Q_bar_t = self.q_bar_at(t)  # (2, 2)
        x_onehot = ops.one_hot(
            x.astype(ms.int32),
            2,
            ms.Tensor(1.0, ms.float32),
            ms.Tensor(0.0, ms.float32),
        )  # (E, 2)
        xt = ops.matmul(x_onehot.astype(ms.float32), Q_bar_t)  # (E, 2)
        xt = ops.bernoulli(ops.clip_by_value(xt[..., 1], 0.0, 1.0))
        return xt
