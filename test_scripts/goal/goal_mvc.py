import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.goal import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# GOAL Settings
BEAM_SIZE = -1

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mvc/mvc_rb-small_gurobi-60s_205.764.txt",
    "RB-LARGE": "test_dataset/mvc/mvc_rb-large_gurobi-300s_968.228.txt",
    "RB-GIANT": "test_dataset/mvc/mvc_rb-giant_gurobi-3600s_2396.780.txt",
    "TWITTER": "test_dataset/mvc/mvc_twitter_gurobi-60s_85.251.txt",
    "COLLAB": "test_dataset/mvc/mvc_collab_gurobi-60s_65.086.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/goal/goal_mvc_rb-small.pt",
    "RB-LARGE": "weights/goal/goal_mvc_rb-large.pt",
    "RB-GIANT": "weights/goal/goal_mvc_rb-large.pt",
    "TWITTER": "weights/goal/goal_mvc_rb-small.pt",
    "COLLAB": "weights/goal/goal_mvc_rb-small.pt"
}

# main
if __name__ == "__main__":
    solver = GOALMVCSolver(
        model=GOALModel(
            env=GOALEnv(task="MVC", device="cuda"),
            encoder=GATEncoder(task="MVC"),
            decoder=GOALDecoder(),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
        ),
        beam_size=BEAM_SIZE
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True, check_constraint=False))