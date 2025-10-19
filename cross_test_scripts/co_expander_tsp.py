import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "Cluster-500"
SPARSE_FACTOR = 50 # 50 for TSP-500 and TSP-10K, 100 for TSP-1K

# COExpander Settings
INFERENCE_STEP = 5 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 4 # S

# Local Search Settings
USE_2OPT = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "Cluster-500": "cross_dataset/tsp/tsp500_cluster_10.723.txt",
    "Gaussian-500": "cross_dataset/tsp/tsp500_gaussian_77.521.txt",
}
WEIGHT_PATH_DICT = {
    "Cluster-500": "weights/co_expander/coexpander_tsp500_sparse.pt",
    "Gaussian-500": "weights/co_expander/coexpander_tsp500_sparse.pt",
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
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True, normalize=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))