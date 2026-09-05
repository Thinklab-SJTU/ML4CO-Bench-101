import numpy as np
import mindspore as ms
from typing import Union
from mindspore import Tensor


ArrayLike = Union[np.ndarray, Tensor, list]


def to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Tensor):
        return x.asnumpy()
    if isinstance(x, list):
        return np.asarray(x)
    raise TypeError(f"Unsupported type for to_numpy: {type(x)}")


def to_tensor(x: ArrayLike, dtype=None) -> Tensor:
    """Convert numpy / list / Tensor to ``ms.Tensor``."""
    if isinstance(x, Tensor):
        return x.astype(dtype) if dtype is not None else x
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if dtype is None:
            return ms.Tensor(np.ascontiguousarray(x))
        return ms.Tensor(np.ascontiguousarray(x), dtype=dtype)
    raise TypeError(f"Unsupported type for to_tensor: {type(x)}")
