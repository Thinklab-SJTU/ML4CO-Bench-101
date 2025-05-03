import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
NODES_NUM = 50
SPARSE_FACTOR = -1

# COExpander Settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_LOCAL_SEARCH = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt",
    100: "test_dataset/cvrp/cvrp100_hgs-20s_15.563.txt",
    200: "test_dataset/cvrp/cvrp200_hgs-60s_19.630.txt",
    500: "test_dataset/cvrp/cvrp500_hgs-300s_37.154.txt",
}
WEIGHT_PATH_DICT = {
    50: "weights/co_expander/coexpander_cvrp50_dense.pt",
    100: "weights/co_expander/coexpander_cvrp100_dense.pt",
    200: "weights/co_expander/coexpander_cvrp200_dense.pt",
    500: "weights/co_expander/coexpander_cvrp500_dense.pt",
}

# Main
if __name__ == "__main__":
    solver = COExpanderCVRPSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="CVRP", sparse_factor=-1, device="cuda"
            ),
            encoder=GNNEncoder(
                task="CVRP",
                sparse=False,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=COExpanderDecoder(
                decode_kwargs={"use_ls": USE_LOCAL_SEARCH}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))