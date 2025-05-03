import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.matnet import *

# Problem Settings
NODES_NUM = 50

# Solving Settings
BEAM_SIZE = -1

# Local Search Settings
USE_2OPT = False

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/atsp/atsp50_uniform_lkh_1000_1.5545.txt",
    100: "test_dataset/atsp/atsp100_uniform_lkh_1000_1.5660.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/matnet/matnet_atsp50.pt",
    100: "weights/matnet/matnet_atsp100.pt"
}

# Main
if __name__ == "__main__":
    solver = MatNetATSPSolver(
        model=MatNetATSPModel(
            env=MatNetATSPEnv(
                nodes_num=NODES_NUM, pomo_size=NODES_NUM, device="cuda"
            ),
            one_hot_seed_cnt=NODES_NUM,
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
        ),
        beam_size=BEAM_SIZE,
        use_2opt=USE_2OPT
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))