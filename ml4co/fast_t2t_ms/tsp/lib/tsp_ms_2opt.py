import mindspore as ms
from mindspore import Tensor
from ml4co.fast_t2t_ms.common.type_utils import to_numpy, to_tensor
from .tsp_2opt import c_tsp_2opt


def mindspore_tsp_2opt(
    points: Tensor,
    tours: Tensor,
    max_iters: int = 5000,
    type_2opt: int = 2,
    num_workers: int = 1,
) -> Tensor:
    """
    Args:
        points: (B, N, 2) float
        tours: (B, N+1) int closed tours
    Returns:
        Optimized tours as ``ms.Tensor`` int32.
    """
    np_points = to_numpy(points).astype("float32")
    np_tours = to_numpy(tours).astype("int32")
    if np_points.ndim == 2:
        np_points = np_points[None, ...]
    out = c_tsp_2opt(
        points=np_points,
        tours=np_tours,
        max_iters=max_iters,
        type_2opt=type_2opt,
        num_workers=num_workers,
    )
    return to_tensor(out, dtype=ms.int32)
