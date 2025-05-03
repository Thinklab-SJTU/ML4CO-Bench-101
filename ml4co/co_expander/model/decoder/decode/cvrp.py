import copy
import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import (
    to_numpy, CVRPSolver, cvrp_classic_local_search, cvrp_greedy_decoder
)


def cvrp_dense_decode(
    vars: Tensor, nodes_feature: Tensor, return_cost: bool, use_ls: bool = False
) -> Union[List[np.ndarray], np.floating]:
    # heatmap
    heatmap: np.ndarray = to_numpy(vars)
    nodes_feature: np.ndarray = to_numpy(nodes_feature)
    depots = nodes_feature[:, 0, :2]
    points = nodes_feature[:, 1:, :2]
    demands = nodes_feature[:, :, 2]
    heatmap = (heatmap + heatmap.transpose(0, 2, 1)) / 2

    # decoding 
    tours = list()
    for idx in range(len(heatmap)):
        tour = cvrp_greedy_decoder(
            heatmap=copy.deepcopy(heatmap[idx]), norm_demand=demands[idx]
        )
        if use_ls:
            tour = cvrp_classic_local_search(
                init_tour=tour, depot=depots[idx],
                points=points[idx], demands=demands[idx]
            )
        tours.append(tour)
    if return_cost:
        tmp_solver = CVRPSolver()
        tmp_solver.from_data(
            depots=depots, 
            points=points, 
            demands=demands[:, 1:], 
            capacities=np.ones(shape=(len(heatmap),)),
            tours=tours
        )
        return tmp_solver.evaluate()
    else:
        return tours