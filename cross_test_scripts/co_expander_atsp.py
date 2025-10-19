import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "HCP-50"
SPARSE_FACTOR = -1

# COExpander Settings
INFERENCE_STEP = 5 # Is
DETERMINATE_STEP = 3 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_2OPT = True

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "HCP-50": "cross_dataset/atsp/atsp50_hcp_0.000.txt",
    "SAT-54": "cross_dataset/atsp/atsp54_sat_0.000.txt",
    "HCP-100": "cross_dataset/atsp/atsp100_hcp_0.000.txt",
    "SAT-102": "cross_dataset/atsp/atsp102_sat_0.000.txt",
    "HCP-200": "cross_dataset/atsp/atsp200_hcp_0.000.txt",
    "SAT-200": "cross_dataset/atsp/atsp200_sat_0.000.txt",
    "HCP-500": "cross_dataset/atsp/atsp500_hcp_0.000.txt",
    "SAT-507": "cross_dataset/atsp/atsp507_sat_0.000.txt"
}
WEIGHT_PATH_DICT = {
    "HCP-50": "weights/co_expander/coexpander_atsp50_dense.pt",
    "SAT-54": "weights/co_expander/coexpander_atsp50_dense.pt",
    "HCP-100": "weights/co_expander/coexpander_atsp100_dense.pt",
    "SAT-102": "weights/co_expander/coexpander_atsp100_dense.pt",
    "HCP-200": "weights/co_expander/coexpander_atsp200_dense.pt",
    "SAT-200": "weights/co_expander/coexpander_atsp200_dense.pt",
    "HCP-500": "weights/co_expander/coexpander_atsp500_dense.pt",
    "SAT-507": "weights/co_expander/coexpander_atsp500_dense.pt"
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
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))