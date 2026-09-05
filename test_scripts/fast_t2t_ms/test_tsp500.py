import os
import sys
root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
from ml4co.fast_t2t_ms.skip_ortools import apply as skip_ortools
skip_ortools()
from ml4co_kit import TSPWrapper
from test_scripts.test_dataset import TSP500_TEST_PATH
from ml4co.fast_t2t_ms import TSPModel, TSPEnv, TSPPLModel, FastT2TMSSolver


# Settings
DEVICE = "CPU"
DEVICE_ID = 0
TEST_DATA_PATH = TSP500_TEST_PATH
WEIGHT_PATH = f"weights/fast_t2t_ms/tsp500_fast_t2t.ckpt"
SOLVE_STEPS = 1
RUNS_NUM = 1
BATCH_SIZE = 1


# Main
if __name__ == "__main__":
    env = TSPEnv(mode="solve", device=DEVICE, device_id=DEVICE_ID)

    model = TSPModel(hidden_dim=256, num_layers=12)

    pl_model = TSPPLModel(
        env=env,
        model=model,
        weight_path=WEIGHT_PATH,
        cm_steps=SOLVE_STEPS
    )

    wrapper = TSPWrapper()
    if TEST_DATA_PATH.endswith(".txt"):
        wrapper.from_txt(TEST_DATA_PATH, ref=True)
    elif TEST_DATA_PATH.endswith(".pkl"):
        wrapper.from_pickle(TEST_DATA_PATH)
    else:
        raise ValueError(f"Unsupported file type")

    solver = FastT2TMSSolver(
        pl_model=pl_model, seed=1234, runs_num=RUNS_NUM
    )

    wrapper.solve(solver, batch_size=BATCH_SIZE, show_time=True)

    print(wrapper.evaluate_w_gap())
