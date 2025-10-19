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
INFERENCE_STEP = 5 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_2OPT = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/atsp/atsp50_uniform_lkh_1000_1.5545.txt",
    100: "test_dataset/atsp/atsp100_uniform_lkh_1000_1.5660.txt",
    200: "test_dataset/atsp/atsp200_uniform_lkh_1000_1.5647.txt",
    500: "test_dataset/atsp/atsp500_uniform_lkh_1000_1.5734.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/co_expander/coexpander_atsp50_dense.pt",
    100: "weights/co_expander/coexpander_atsp100_dense.pt",
    200: "weights/co_expander/coexpander_atsp200_dense.pt",
    500: "weights/co_expander/coexpander_atsp500_dense.pt"
}

# Main
if __name__ == "__main__":
    solver = COExpanderATSPSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="ATSP", sparse_factor=SPARSE_FACTOR, device="cuda"
            ),
            encoder=GNNEncoder(
                task="ATSP", sparse=SPARSE_FACTOR>0, block_layers=[2, 4, 4, 2]
            ),
            decoder=COExpanderDecoder(
                decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))