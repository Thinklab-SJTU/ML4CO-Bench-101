import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.symnco import *

# Problem Settings
NODES_NUM = 50

# Solving Settings
DECODE_TYPE = "multistart_greedy" # "multistart_greedy"

# Local Search Settings
USE_TWO_OPT = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/tsp/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp/tsp100_concorde_7.756.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/symnco/symnco_tsp50.pt",
    100: "weights/symnco/symnco_tsp100.pt",
}

# Main
if __name__ == "__main__":
    solver = SymNCOTSPSolver(
        model=SymNCOModel(
            env=SymNCOTSPEnv(device="cuda"),
            encoder=GATEncoder(task="TSP", hidden_dim=128, num_layers=3),
            decoder=SymNCODecoder(task="TSP", hidden_dim=128),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            decode_type=DECODE_TYPE
        ),
        use_2opt=USE_TWO_OPT
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))