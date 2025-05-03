import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# COExpander Settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 20 # Ds
SAMPLING_NUM = 1 # S

# Local Search Settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-LARGE": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "TWITTER": (0.01, 2, 1000, 200, 4.0, 0.2),
    "COLLAB": (0.01, 2, 1000, 300, 1.02, 0.2)
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mvc/mvc_rb-small_gurobi-60s_205.764.txt",
    "RB-LARGE": "test_dataset/mvc/mvc_rb-large_gurobi-300s_968.228.txt",
    "RB-GIANT": "test_dataset/mvc/mvc_rb-giant_gurobi-3600s_2396.780.txt",
    "TWITTER": "test_dataset/mvc/mvc_twitter_gurobi-60s_85.251.txt",
    "COLLAB": "test_dataset/mvc/mvc_collab_gurobi-60s_65.086.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/co_expander/coexpander_mvc_rb-small_sparse.pt",
    "RB-LARGE": "weights/co_expander/coexpander_mvc_rb-large_sparse.pt",
    "RB-GIANT": "weights/co_expander/coexpander_mvc_rb-large_sparse.pt",
    "TWITTER": "weights/co_expander/coexpander_mvc_rb-small_sparse.pt",
    "COLLAB": "weights/co_expander/coexpander_mvc_rb-small_sparse.pt"
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = COExpanderMVCSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="MVC", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MVC",
                sparse=True,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=COExpanderDecoder(
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
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))