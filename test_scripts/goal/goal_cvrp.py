import os
import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.goal import *

# Problem Settings
NODES_NUM = 50

# GOAL Settings
BEAM_SIZE = -1

# Local Search Settings
USE_LOCAL_SEARCH = False

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt",
    100: "test_dataset/cvrp/cvrp100_hgs-20s_15.563.txt",
    200: "test_dataset/cvrp/cvrp200_hgs-60s_19.630.txt",
    500: "test_dataset/cvrp/cvrp500_hgs-300s_37.154.txt",
}
WEIGHT_PATH_DICT = {
    50: "weights/goal/goal_cvrp50.pt",
    100: "weights/goal/goal_cvrp100.pt",
    200: "weights/goal/goal_cvrp200.pt",
    500: "weights/goal/goal_cvrp500.pt",
}

# Main
if __name__ == "__main__":
    solver = GOALCVRPSolver(
        model=GOALModel(
            env=GOALEnv(task="CVRP", device="cuda"),
            encoder=GATEncoder(task="CVRP"),
            decoder=GOALDecoder(
                decode_kwargs={"use_ls": USE_LOCAL_SEARCH}
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM],
        ),
        beam_size=BEAM_SIZE
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))