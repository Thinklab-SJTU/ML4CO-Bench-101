import os
import sys
root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
from ml4co_kit import TSPWrapper
from test_scripts.test_dataset import TSP100_TEST_PATH
from ml4co.fast_t2t import TSPModel, TSPEnv, TSPPLModel, FastT2TSolver


# Settings
DEVICE = "cuda"
TEST_DATA_PATH = TSP100_TEST_PATH
WEIGHT_PATH = f"weights/fast_t2t/tsp100_fast_t2t.pt"
SOLVE_STEPS = 1
RUNS_NUM = 1
BATCH_SIZE = 1


# Main
if __name__ == "__main__":
    # Create Environment
    env = TSPEnv(mode="solve", device="cuda")

    # Create Model
    model = TSPModel(hidden_dim=256, num_layers=12)

    # Create PL Model
    pl_model = TSPPLModel(
        env=env, 
        model=model,
        weight_path=WEIGHT_PATH,
        cm_steps=SOLVE_STEPS
    )

    # Create Wrapper and read test data
    wrapper = TSPWrapper()
    if TEST_DATA_PATH.endswith(".txt"):
        wrapper.from_txt(TEST_DATA_PATH, ref=True)
    elif TEST_DATA_PATH.endswith(".pkl"):
        wrapper.from_pickle(TEST_DATA_PATH)
    else:
        raise ValueError(f"Unsupported file type")
    
    # Create Solver
    solver = FastT2TSolver(
        pl_model=pl_model, seed=1234, runs_num=RUNS_NUM
    )

    # Solve
    wrapper.solve(solver, batch_size=BATCH_SIZE, show_time=True)

    # Evaluate
    print(wrapper.evaluate_w_gap())