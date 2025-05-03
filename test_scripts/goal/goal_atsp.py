import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.goal import *

# Problem Settings
NODES_NUM = 200
KNNS = -1

# GOAL Settings
BEAM_SIZE = 16

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
    50: "weights/goal/goal_atsp50.pt",
    100: "weights/goal/goal_atsp100.pt",
    200: "weights/goal/goal_atsp200.pt",
    500: "weights/goal/goal_atsp500.pt"
}

# Main
if __name__ == "__main__":
    solver = GOALATSPSolver(
        model=GOALModel(
            env=GOALEnv(task="ATSP", device="cuda"),
            encoder=GATEncoder(task="ATSP"),
            decoder=GOALDecoder(
                decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
        ),
        beam_size=BEAM_SIZE
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))