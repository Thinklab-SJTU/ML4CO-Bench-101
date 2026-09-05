import random
import numpy as np
from typing import List
from ml4co_kit import SOLVER_TYPE, SolverBase, TaskBase
from ml4co.fast_t2t_ms.common import MetaPLModel


class FastT2TMSSolver(SolverBase):
    def __init__(
        self,
        pl_model: MetaPLModel,
        seed: int = 1234,
        runs_num: int = 1,
    ):
        # Super Args
        super(FastT2TMSSolver, self).__init__(
            solver_type=SOLVER_TYPE.DIY
        )

        # Attributes
        self.pl_model = pl_model
        self.pl_model.set_train(False)
        for p in self.pl_model.get_parameters():
            p.requires_grad = False
        self.runs_num = runs_num
        self.set_seed(seed)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def _solve(self, task_data: TaskBase):
        return self._batch_solve([task_data])

    def _batch_solve(self, batch_task_data: List[TaskBase]):
        # Process batch data
        batch_task_data, batch_processed_data = \
        self.pl_model.env.process_batch_data(
            batch_task_data=batch_task_data,
            runs_num=self.runs_num,
        )

        # Solve
        self.pl_model._solve_step(
            batch_task_data=batch_task_data,
            batch_processed_data=batch_processed_data,
            solve_runs=self.runs_num,
        )
