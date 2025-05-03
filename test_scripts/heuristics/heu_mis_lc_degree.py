import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.heuristics.mis_lc_degree import MISLCDegreeSolver

# basic settings
TEST_TYPE = "RB-GIANT"

# rlsa settings
USE_RLSA = True
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 1000, 100, 1.02),
    "RB-LARGE": (0.01, 2, 1000, 2000, 1.02),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02),
    "ER-700-800": (0.2, 5, 1000, 300, 1.001),
    "ER-1400-1600": (0.2, 5, 2000, 1000, 1.001),
    "SATLIB": (0.2, 5, 1000, 1.02),
}

# test files
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mis/mis_rb-small_kamis-60s_20.090.txt",
    "RB-LARGE": "test_dataset/mis/mis_rb-large_kamis-60s_43.004.txt",
    "RB-GIANT": "test_dataset/mis/mis_rb-giant_kamis-60s_49.260.txt",
    "ER-700-800": "test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt",
    "ER-1400-1600": "test_dataset/mis/mis_er-1400-1600_kamis-60s_50.938.txt",
    "SATLIB": "test_dataset/mis/mis_satlib_kamis-60s_425.954.txt"
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = MISLCDegreeSolver(
        use_rlsa=USE_RLSA,
        rlsa_tau=rlsa_settings[0],
        rlsa_d=rlsa_settings[1],
        rlsa_k=rlsa_settings[2],
        rlsa_t=rlsa_settings[3],
        rlsa_beta=rlsa_settings[4],
        rlsa_device="cuda"
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))