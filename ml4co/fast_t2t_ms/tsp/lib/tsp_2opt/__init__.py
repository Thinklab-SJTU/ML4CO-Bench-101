import os
import shutil
import pathlib
import numpy as np

try:
    from .source import tsp_2opt_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import tsp_2opt_impl

c_tsp_2opt_impl = tsp_2opt_impl.tsp_2opt


def c_tsp_2opt(
    points: np.ndarray,
    tours: np.ndarray,
    max_iters: int = 5000,
    type_2opt: int = 2,
    num_workers: int = 1,
) -> np.ndarray:
    """
    Exact pairwise 2-opt (Euclidean), from RS4CO ``tsp_ops.hpp``.

    Args:
        points: (B, N, 2) float32 coordinates
        tours: (B, N+1) int32 closed tours
        max_iters: max 2-opt iterations per instance
        type_2opt: 1 = first improving, 2 = best improving
        num_workers: parallel workers over the batch

    Returns:
        Optimized tours (B, N+1) int32; input unchanged.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    t = np.ascontiguousarray(tours, dtype=np.int32)
    return c_tsp_2opt_impl(pts, t, int(max_iters), int(type_2opt), int(num_workers))
