"""MindSpore ops helpers with Ascend-safe fallbacks."""

from __future__ import annotations

import mindspore as ms
from mindspore import Tensor, ops


def bernoulli(probs: Tensor) -> Tensor:
    """
    Sample Bernoulli(probs) as float {0, 1}.

    Ascend (MindSpore 2.7) has no Bernoulli OpAdapter:
    ``Can't find OpAdapter for Bernoulli``. Use Uniform + compare instead,
    which matches ``torch.bernoulli`` / ``ops.bernoulli`` semantics.
    """
    p = ops.clip_by_value(probs.astype(ms.float32), 0.0, 1.0)
    u0 = ms.Tensor(0.0, ms.float32)
    u1 = ms.Tensor(1.0, ms.float32)
    u = ops.uniform(p.shape, u0, u1)
    return (u < p).astype(ms.float32)
