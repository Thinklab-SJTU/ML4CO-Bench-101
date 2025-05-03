import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.heuristics.mcl_rlsa import MClRLSASolver

# basic settings
TEST_TYPE = "RB-LARGE"

# rlsa settings
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 100, 1.02),
    "RB-LARGE": (0.01, 2, 500, 1.02),
    "RB-GIANT": (0.01, 2, 500, 1.02),
    "TWITTER": (0.01, 2, 50, 4.0),
    "COLLAB": (0.01, 2, 50, 1.001),
}

# test files
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mcl/mcl_rb-small_gurobi-60s_19.082.txt",
    "RB-LARGE": "test_dataset/mcl/mcl_rb-large_gurobi-300s_40.182.txt",
    "RB-GIANT": "test_dataset/mcl/mcl_rb-giant_gurobi-3600s_81.520.txt",
    "TWITTER": "test_dataset/mcl/mcl_twitter_gurobi-60s_14.210_new.txt",
    "COLLAB": "test_dataset/mcl/mcl_collab_gurobi-60s_42.113_new.txt"
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = MClRLSASolver(
        rlsa_tau=rlsa_settings[0],
        rlsa_d=rlsa_settings[1],
        rlsa_t=rlsa_settings[2],
        rlsa_beta=rlsa_settings[3],
        rlsa_device="cuda"
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))