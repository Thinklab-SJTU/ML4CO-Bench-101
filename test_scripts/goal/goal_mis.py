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
    "RB-SMALL": "test_dataset/mis/mis_rb-small_kamis-60s_20.090.txt",
    "RB-LARGE": "test_dataset/mis/mis_rb-large_kamis-60s_43.004.txt",
    "RB-GIANT": "test_dataset/mis/mis_rb-giant_kamis-60s_49.260.txt",
    "ER-700-800": "test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt",
    "ER-1400-1600": "test_dataset/mis/mis_er-1400-1600_kamis-60s_50.938.txt",
    "SATLIB": "test_dataset/mis/mis_satlib_kamis-60s_425.954.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/goal/goal_mis_rb-large.pt",
    "RB-LARGE": "weights/goal/goal_mis_rb-large.pt",
    "RB-GIANT": "weights/goal/goal_mis_rb-large.pt",
    "ER-700-800": "weights/goal/goal_mis_er-700-800.pt",
    "ER-1400-1600": "weights/goal/goal_mis_er-700-800.pt",
    "SATLIB": "weights/goal/goal_mis_satlib.pt"
}

# main
if __name__ == "__main__":
    solver = GOALMISSolver(
        model=GOALModel(
            env=GOALEnv(task="MIS", device="cuda"),
            encoder=GATEncoder(task="MIS"),
            decoder=GOALDecoder(),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
        ),
        beam_size=BEAM_SIZE
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True, check_constraint=False))