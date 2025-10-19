import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
NODES_NUM = 50
SPARSE_FACTOR = -1 # 50 for TSP-500 and TSP-10K, 100 for TSP-1K

# COExpander Settings
INFERENCE_STEP = 5 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_2OPT = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/tsp/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp/tsp500_concorde_16.546.txt",
    1000: "test_dataset/tsp/tsp1000_concorde_23.118.txt",
    10000: "test_dataset/tsp/tsp10000_lkh_500_71.755.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/co_expander/coexpander_tsp50_dense.pt",
    100: "weights/co_expander/coexpander_tsp100_dense.pt",
    500: "weights/co_expander/coexpander_tsp500_sparse.pt",
    1000: "weights/co_expander/coexpander_tsp1k_sparse.pt",
    10000: "weights/co_expander/coexpander_tsp10k_sparse.pt"
}

# Main
if __name__ == "__main__":
    solver = COExpanderTSPSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="TSP", sparse_factor=SPARSE_FACTOR, device="cuda"
            ),
            encoder=TSPGNNEncoder(sparse=SPARSE_FACTOR>0),
            decoder=COExpanderDecoder(
                decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))