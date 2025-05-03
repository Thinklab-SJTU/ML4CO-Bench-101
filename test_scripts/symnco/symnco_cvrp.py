import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.symnco import *

# Problem Settings
NODES_NUM = 100

# Solving Settings
DECODE_TYPE = "greedy" # "multistart_greedy"

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
    50: "weights/symnco/symnco_cvrp50.pt",
    100: "weights/symnco/symnco_cvrp100.pt",
    200: "weights/symnco/symnco_cvrp200.pt",
    500: "weights/symnco/symnco_cvrp500.pt",
}

# Main
if __name__ == "__main__":
    solver = SymNCOCVRPSolver(
        model=SymNCOModel(
            env=SymNCOCVRPEnv(device="cuda"),
            encoder=GATEncoder(task="CVRP", hidden_dim=128, num_layers=3),
            decoder=SymNCODecoder(task="CVRP", hidden_dim=128),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            decode_type=DECODE_TYPE
        ),
        use_ls=USE_LOCAL_SEARCH
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))