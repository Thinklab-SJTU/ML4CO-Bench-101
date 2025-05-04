import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.goal import *

# Problem Settings
NODES_NUM = 50
KNNS = -1

# Solving Settings
BEAM_SIZE = -1

# Local Search Settings
USE_2OPT = False

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/tsp/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp/tsp500_concorde_16.546.txt",
    1000: "test_dataset/tsp/tsp1000_concorde_23.118.txt",
    10000: "test_dataset/tsp/tsp10000_lkh_500_71.755.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/goal/goal_tsp50.pt",
    100: "weights/goal/goal_tsp100.pt",
    500: "weights/goal/goal_tsp500.pt",
    1000: "weights/goal/goal_tsp1k.pt",
    10000: "weights/goal/goal_tsp1k.pt"
}

# Main
if __name__ == "__main__":
    solver = GOALTSPSolver(
        model=GOALModel(
            env=GOALEnv(task="TSP", device="cuda"),
            encoder=GATEncoder(task="TSP"),
            decoder=GOALDecoder(
                knns=KNNS, decode_kwargs={"use_2opt": USE_2OPT}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
        ),
        beam_size=BEAM_SIZE
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))