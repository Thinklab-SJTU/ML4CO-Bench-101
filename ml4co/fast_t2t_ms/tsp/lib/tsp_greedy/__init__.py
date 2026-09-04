import os
import shutil
import pathlib
import numpy as np

try:
    from .source import tsp_greedy_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import tsp_greedy_impl

c_tsp_greedy_impl = tsp_greedy_impl.tsp_greedy_insert


def c_tsp_greedy(
    top_edges: np.ndarray, 
    nodes_num: int, 
    num_workers: int = 1
) -> np.ndarray:
    return c_tsp_greedy_impl(top_edges, nodes_num, num_workers)