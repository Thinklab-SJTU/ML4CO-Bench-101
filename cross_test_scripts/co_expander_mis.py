import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "HK-SMALL"

# COExpander Settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 20 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

# Local Search Settings
USE_RLSA = True
RLSA_SETTINGS_DICT = {
    "HK-SMALL": (0.01, 5, 1000, 500, 1.02, 0.3),
    "HK-LARGE": (0.01, 5, 1000, 1000, 1.02, 0.3),
    "WS-SMALL": (0.01, 5, 1000, 500, 1.02, 0.3),
    "WS-LARGE": (0.01, 5, 1000, 1000, 1.02, 0.3),
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "HK-SMALL": "cross_dataset/mis/mis_hk_small_query.txt",
    "HK-LARGE": "cross_dataset/mis/mis_hk_large_query.txt",
    "WS-SMALL": "cross_dataset/mis/mis_ws_small_query.txt",
    "WS-LARGE": "cross_dataset/mis/mis_ws_large_query.txt",
}
WEIGHT_PATH_DICT = {
    "HK-SMALL": "weights/co_expander/coexpander_mis_er-700-800_sparse.pt",
    "HK-LARGE": "weights/co_expander/coexpander_mis_er-700-800_sparse.pt",
    "WS-SMALL": "weights/co_expander/coexpander_mis_er-700-800_sparse.pt",
    "WS-LARGE": "weights/co_expander/coexpander_mis_er-700-800_sparse.pt",
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = COExpanderMISSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="MIS", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MIS",
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
            determinate_steps=DETERMINATE_STEP,
            beam_size=BEAM_SIZE,
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))