import torch
from torch import Tensor
from typing import Sequence
from dataclasses import dataclass
from ml4co_kit import ATSPDataGenerator


@dataclass
class StepState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor # shape: (batch, pomo)
    current_node: torch.Tensor = None # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None # shape: (batch, pomo, node)


class MatNetATSPEnv:
    def __init__(self, nodes_num, pomo_size: int, device: str = "cpu"):
        self.nodes_num = nodes_num
        self.pomo_size = pomo_size
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.problems = None
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.step_state = None
        self.tours = list()
        self.device = device
        self.generator = ATSPDataGenerator(
            only_instance_for_us=True, nodes_num=nodes_num
        )

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.problems = self.generator.generate_only_instance_for_us(batch_size)

    def load_problems_manual(self, problems: Tensor):
        self.batch_size = problems.shape[0]
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None]
        self.BATCH_IDX = self.BATCH_IDX.expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :]
        self.POMO_IDX = self.POMO_IDX.expand(self.batch_size, self.pomo_size).to(self.device)
        self.problems = problems

    def reset(self):
        self.tours = list()
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.device
        )
        self._create_step_state()
        reward = None
        done = False
        return self.problems, reward, done

    def _create_step_state(self):
        self.step_state = StepState(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.nodes_num)
        ).to(self.device)

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, node_idx) -> Sequence[Tensor]:
        self.tours.append(node_idx)
        self.selected_count += 1
        self.current_node = node_idx
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        self._update_step_state()
        done = (self.selected_count == self.nodes_num)
        if done:
            reward = -self._get_total_distance()
        else:    
            reward = None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')

    def _get_total_distance(self):
        node_from = self.selected_node_list
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(
            self.batch_size, self.pomo_size, self.nodes_num
        )
        selected_cost = self.problems[batch_index, node_from, node_to]
        total_distance = selected_cost.sum(2)

        return total_distance
