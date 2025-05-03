import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.meta_egn import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# Local Search Settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 5, 1000, 500, 1.02, 0.3),
    "RB-LARGE": (0.01, 5, 1000, 1000, 1.02, 0.15),
    "RB-GIANT": (0.01, 5, 1000, 1000, 1.02, 0.1),
    "ER-700-800": (0.2, 10, 1000, 1000, 1.001, 0.2),
    "ER-1400-1600": (0.2, 10, 1000, 1000, 1.001, 0.1),
    "SATLIB": (0.2, 5, 1000, 1000, 1.02, 0.3),
}

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
    "RB-SMALL": "weights/meta_egn/meta_egn_mis_rb-small_sparse.pt",
    "RB-LARGE": "weights/meta_egn/meta_egn_mis_rb-large_sparse.pt",
    "RB-GIANT": "weights/meta_egn/meta_egn_mis_rb-large_sparse.pt",
    "ER-700-800": "weights/meta_egn/meta_egn_mis_er-700-800_sparse.pt",
    "ER-1400-1600": "weights/meta_egn/meta_egn_mis_er-700-800_sparse.pt",
    "SATLIB": "weights/meta_egn/meta_egn_mis_satlib_sparse.pt"
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    processed_path = TEST_FILE_DICT[TEST_TYPE][:-4] + ".dat"
    solver = MetaEGNMISSolver(
        model=MetaEGNModel(
            env=MetaEGNEnv(task="MIS", mode="solve", device="cuda"),
            encoder=GNNEncoder(
                num_layers=6, hidden_dim_1=64, hidden_dim_2=1, devide="cuda"
            ),
            decoder=MetaEGNDecoder(
                decode_kwargs={
                    "use_rlsa": USE_RLSA,
                    "rlsa_tau": rlsa_settings[0],
                    "rlsa_d": rlsa_settings[1],
                    "rlsa_k": rlsa_settings[2],
                    "rlsa_t": rlsa_settings[3],
                    "rlsa_beta": rlsa_settings[4],
                    "rlsa_alpha": rlsa_settings[5],
                }
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
        ),
        processed_path=processed_path
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))


