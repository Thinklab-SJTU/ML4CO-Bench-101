import os
import sys
import numpy as np
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.symnco import *

# Problem Settings
PROBLEMS = [
    ["X-n101-k25", 101],
    ["X-n106-k14", 106],
    ["X-n110-k13", 110],
    ["X-n115-k10", 115],
    ["X-n120-k6", 120],
    ["X-n125-k30", 125],
    ["X-n129-k18", 129],
    ["X-n134-k13", 134],
    ["X-n139-k10", 139],
    ["X-n143-k7", 143],
    ["X-n148-k46", 148],
    ["X-n153-k22", 153],
    ["X-n157-k13", 157],
    ["X-n162-k11", 162],
    ["X-n167-k10", 167],
    ["X-n172-k51", 172],
    ["X-n176-k26", 176],
    ["X-n181-k23", 181],
    ["X-n186-k15", 186],
    ["X-n190-k8", 190],
    ["X-n195-k51", 195],
    ["X-n200-k36", 200],
    ["X-n204-k19", 204],
    ["X-n209-k16", 209],
    ["X-n214-k11", 214],
    ["X-n219-k73", 219],
    ["X-n223-k34", 223],
    ["X-n228-k23", 228],
    ["X-n233-k16", 233],
    ["X-n237-k14", 237],
    ["X-n242-k48", 242],
    ["X-n247-k50", 247],
    ["X-n251-k28", 251],
    ["X-n256-k16", 256],
    ["X-n261-k13", 261],
    ["X-n266-k58", 266],
    ["X-n270-k35", 270],
    ["X-n275-k28", 275],
    ["X-n280-k17", 280],
    ["X-n284-k15", 284],
    ["X-n289-k60", 289],
    ["X-n294-k50", 294],
    ["X-n298-k31", 298],
    ["X-n303-k21", 303],
    ["X-n308-k13", 308],
    ["X-n313-k71", 313],
    ["X-n317-k53", 317],
    ["X-n322-k28", 322],
    ["X-n327-k20", 327],
    ["X-n331-k15", 331],
    ["X-n336-k84", 336],
    ["X-n344-k43", 344],
    ["X-n351-k40", 351],
    ["X-n359-k29", 359],
    ["X-n367-k17", 367],
    ["X-n376-k94", 376],
    ["X-n384-k52", 384],
    ["X-n393-k38", 393],
    ["X-n401-k29", 401],
    ["X-n411-k19", 411],
    ["X-n420-k130", 420],
    ["X-n429-k61", 429],
    ["X-n439-k37", 439],
    ["X-n449-k29", 449],
    ["X-n459-k26", 459],
    ["X-n469-k138", 469],
    ["X-n480-k70", 480],
    ["X-n491-k59", 491],
    ["X-n502-k39", 502],
    ["X-n513-k21", 513],
]

# Solving Settings
DECODE_TYPE = "multistart_sampling"

# Local Search Settings
USE_LOCAL_SEARCH = True

# Test File & Pretrained Settings
WEIGHT_PATH_DICT = {
    50: "weights/goal/goal_cvrp50.pt",
    100: "weights/goal/goal_cvrp100.pt",
    200: "weights/goal/goal_cvrp200.pt",
    500: "weights/goal/goal_cvrp500.pt",
}

# Main
if __name__ == "__main__":
    # init
    gaps = list()
    costs = list()
    
    # nodes num and sparse factor
    for NODES_NUM in [100, 200, 500]:
        for problem in PROBLEMS:
            name, nodes_num = problem
            txt_file = f"test_dataset/cvrp/cvrplib/{name}.txt"
            # solve
            solver = SymNCOCVRPSolver(
                model=SymNCOModel(
                    env=SymNCOCVRPEnv(device="cuda"),
                    encoder=GATEncoder(task="CVRP", hidden_dim=128, num_layers=3),
                    decoder=SymNCODecoder(task="CVRP", hidden_dim=128),
                    weight_path=WEIGHT_PATH_DICT[NODES_NUM],
                    decode_type=DECODE_TYPE
                ),
                use_ls=USE_LOCAL_SEARCH
            )
            solver.from_txt(txt_file, ref=True, show_time=False)
            solver.solve(show_time=False)
            result = solver.evaluate(calculate_gap=True)
            print(f"{name}: ", result, flush=True)
            costs.append(result[0])
            gaps.append(result[2])
        np.save(f"gaps_{NODES_NUM}.npy", np.array(gaps))
        np.save(f"costs_{NODES_NUM}.npy", np.array(costs))