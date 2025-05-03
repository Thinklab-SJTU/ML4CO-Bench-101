import torch
import numpy as np
from torch import Tensor
from tensordict.tensordict import TensorDict
from ml4co_kit import TSPDataGenerator, TSPSolver
from symnco.env.base import SymNCOEnv
from symnco.utils.ops import gather_by_index, get_tour_length


class SymNCOTSPEnv(SymNCOEnv):
    batch_locked = False
    def __init__(
        self,
        generator: TSPDataGenerator = None,
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
        super(SymNCOTSPEnv, self).__init__(
            task="TSP",
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
        self.generator: TSPDataGenerator
        self.tmp_solver = TSPSolver()
        self.val_data_cache = None
        
    def generate_train_data(self, batch_size: int) -> TensorDict:
        points = self.generator.generate_only_instance_for_us(batch_size)
        points = torch.from_numpy(points)
        train_data = TensorDict({"locs": points}, batch_size=batch_size, device=self.device)
        return train_data

    def generate_val_data(self, val_idx: int) -> TensorDict:
        # begin and end index
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        
        # check cache
        if self.val_data_cache is None:
            self.tmp_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {"points": self.tmp_solver.points}
            
        # get data
        points = self.val_data_cache["points"][begin_idx:end_idx]
        points = torch.from_numpy(points)
        val_data = TensorDict(
            {"locs": points}, batch_size=self.val_batch_size, device=self.device
        )
        return val_data

    def _step(self, td: TensorDict) -> TensorDict:
        current_node: Tensor = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: TensorDict, batch_size: int = None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs: Tensor = td["locs"]

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros(size=(*batch_size,)).int().to(self.device)
        available = torch.ones(
            size=(*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1)).int().to(self.device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    def _get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)
    
    def process_data(self, batch_size: int, points: np.ndarray) -> TensorDict:
        points = torch.from_numpy(points)
        data = TensorDict(
            {"locs": points}, batch_size=batch_size, device=self.device
        )
        return data