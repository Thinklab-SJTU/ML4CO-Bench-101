import torch
import numpy as np
from torch import Tensor
from tensordict.tensordict import TensorDict
from ml4co_kit import CVRPDataGenerator, CVRPSolver
from symnco.env.base import SymNCOEnv
from symnco.utils.ops import gather_by_index, get_tour_length


class SymNCOCVRPEnv(SymNCOEnv):
    batch_locked = False
    def __init__(
        self,
        generator: CVRPDataGenerator = None,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        device: str = "cpu",
        val_path: str = None,
    ):
        # super args
        super(SymNCOCVRPEnv, self).__init__(
            task="CVRP",
            generator=generator,
            mode=mode,
            train_data_size = train_data_size,
            val_data_size = val_data_size,
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size,
            num_workers = num_workers,
            device = device,
            val_path = val_path,
        )
        self.generator: CVRPDataGenerator
        self.tmp_solver = CVRPSolver()
        self.val_data_cache = None
        
    def generate_train_data(self, batch_size: int) -> TensorDict:
        depots, points, demands, capacities = \
            self.generator.generate_only_instance_for_us(batch_size)
        demands = demands / capacities.reshape(-1, 1)
        train_data = TensorDict(
            {
                "depot": torch.from_numpy(depots).float(),
                "locs": torch.from_numpy(points).float(),
                "demand": torch.from_numpy(demands).float()
            }, 
            batch_size=batch_size, device=self.device
        )
        return train_data

    def generate_val_data(self, val_idx: int) -> TensorDict:
        # begin and end index
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        
        # check cache
        if self.val_data_cache is None:
            self.tmp_solver.from_txt(self.val_path, ref=True)
            demands = self.tmp_solver.demands
            capacities = self.tmp_solver.capacities
            demands = demands / capacities.reshape(-1, 1)
            self.val_data_cache = {
                "depots": self.tmp_solver.depots,
                "points": self.tmp_solver.points,
                "demands": demands,
                "data_size": self.tmp_solver.depots.shape[0]
            }
            
        # get data
        depots = self.val_data_cache["depots"][begin_idx:end_idx]
        points = self.val_data_cache["points"][begin_idx:end_idx]
        demands = self.val_data_cache["demands"][begin_idx:end_idx]
        val_data = TensorDict(
            {
                "depot": torch.from_numpy(depots).float(),
                "locs": torch.from_numpy(points).float(),
                "demand": torch.from_numpy(demands).float()
            }, 
            batch_size=self.val_batch_size, device=self.device
        )
        return val_data

    def _step(self, td: TensorDict) -> TensorDict:
        current_node: Tensor = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot
        
        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )
        
        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)
        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"][:, None, :] + td["used_capacity"][..., None] > 1.0

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

    def _reset(self, td: TensorDict, batch_size: int = None) -> TensorDict:
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), 1.0, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    def _check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are not visited twice except depot and capacity is not exceeded"""
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td["demand"].size()
        sorted_pi = actions.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat((-td["vehicle_capacity"], td["demand"]), 1)
        d = demand_with_depot.gather(1, actions)

        used_cap = torch.zeros_like(td["demand"][:, 0])
        for i in range(actions.size(1)):
            used_cap += d[
                :, i
            ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (
                used_cap <= td["vehicle_capacity"] + 1e-5
            ).all(), "Used more than capacity"

    def _get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)
    
    def process_data(
        self, batch_size: int, depots: np.ndarray, points: np.ndarray,
        demands: np.ndarray, capacities: np.ndarray
    ) -> TensorDict:
        demands = demands / capacities.reshape(-1, 1)
        data = TensorDict(
            {
                "depot": torch.from_numpy(depots).float(),
                "locs": torch.from_numpy(points).float(),
                "demand": torch.from_numpy(demands).float()
            }, 
            batch_size=batch_size, device=self.device
        )
        return data