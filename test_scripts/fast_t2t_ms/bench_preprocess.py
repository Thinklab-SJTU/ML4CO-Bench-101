"""
Quick preprocess vs solve timing (TSP100).

  python test_scripts/fast_t2t_ms/bench_preprocess.py --device Ascend
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ml4co.ms_utils import setup_ld_library_path, skip_ortools

setup_ld_library_path()
skip_ortools()

from ml4co_kit import TSPWrapper
from test_scripts.test_dataset import TSP100_TEST_PATH
from ml4co.fast_t2t_ms import TSPModel, TSPEnv, TSPPLModel
from ml4co.ms_utils import tensor_device_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="Ascend")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--weight", default="weights/fast_t2t_ms/tsp100_fast_t2t.ckpt"
    )
    p.add_argument("--data", default=TSP100_TEST_PATH)
    args = p.parse_args()

    weight = args.weight if os.path.isabs(args.weight) else os.path.join(ROOT, args.weight)
    data = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)

    env = TSPEnv(mode="solve", device=args.device, device_id=args.device_id)
    model = TSPModel(hidden_dim=256, num_layers=12)
    pl = TSPPLModel(
        env=env,
        model=model,
        weight_path=weight if os.path.isfile(weight) else None,
        cm_steps=1,
    )

    wrapper = TSPWrapper()
    wrapper.from_txt(data, ref=True)
    tasks = wrapper.task_list[: args.batch_size]

    # Warmup
    bt, batch = env.process_batch_data(tasks, runs_num=1)
    pl._solve_step(bt, batch, solve_runs=1)

    proc_times, move_times, solve_times = [], [], []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        bt, batch = env.process_batch_data(tasks, runs_num=1)
        t1 = time.perf_counter()
        pl._maybe_to_device(batch)
        t2 = time.perf_counter()
        # resolve again with fresh batch (includes to_device inside)
        bt, batch = env.process_batch_data(tasks, runs_num=1)
        t3 = time.perf_counter()
        pl._solve_step(bt, batch, solve_runs=1)
        t4 = time.perf_counter()
        proc_times.append(t1 - t0)
        move_times.append(t2 - t1)
        solve_times.append(t4 - t3)

    print(f"device={args.device} batch={args.batch_size} repeats={args.repeats}")
    print(
        f"process_batch_data  mean={sum(proc_times)/len(proc_times):.4f}s  "
        f"min={min(proc_times):.4f}s"
    )
    print(
        f"batch.to_device      mean={sum(move_times)/len(move_times):.4f}s  "
        f"min={min(move_times):.4f}s"
    )
    print(
        f"_solve_step         mean={sum(solve_times)/len(solve_times):.4f}s  "
        f"min={min(solve_times):.4f}s"
    )
    print(
        "tensor devices after move: "
        f"points={tensor_device_target(batch.node_feature)} "
        f"ptr={tensor_device_target(batch.ptr)}"
    )
    print(
        "Expect: process << 0.1s, solve ~0.02s on Ascend after warmup. "
        "If process still ~1s, Ascend ops are still in the preprocess path."
    )


if __name__ == "__main__":
    main()
