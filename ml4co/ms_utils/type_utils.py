import numpy as np
import mindspore as ms
from typing import Optional, Union
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


def to_tensor(
    x: ArrayLike,
    dtype=None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Convert numpy / list / Tensor to ``ms.Tensor``.

    If ``device`` is set, place storage there. On Ascend, bare
    ``ms.Tensor(np)`` stays on CPU — we ``move_to`` after construction.
    """
    if isinstance(x, Tensor):
        out = x.astype(dtype) if dtype is not None else x
    elif isinstance(x, list):
        x = np.asarray(x)
        out = (
            ms.Tensor(np.ascontiguousarray(x), dtype)
            if dtype is not None
            else ms.Tensor(np.ascontiguousarray(x))
        )
    elif isinstance(x, np.ndarray):
        out = (
            ms.Tensor(np.ascontiguousarray(x), dtype)
            if dtype is not None
            else ms.Tensor(np.ascontiguousarray(x))
        )
    else:
        raise TypeError(f"Unsupported type for to_tensor: {type(x)}")

    if device is None:
        return out
    from ml4co.ms_utils.device_utils import maybe_move_tensor

    return maybe_move_tensor(out, device, strict=True)
