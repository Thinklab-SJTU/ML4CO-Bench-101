import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Union
from ml4co_kit import BaseModel, TaskBase
from .meta_env import MetaEnv
from .meta_dataset import MetaDataBatch


class MetaPLModel(BaseModel):
    def __init__(
        self,
        env: MetaEnv,
        model: nn.Module,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None,
        cm_alpha: float = 0.2,
        cm_steps: int = 5,
    ):
        # Super Args
        super(MetaPLModel, self).__init__(
            env=env,
            model=model,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # Hint
        self.env: MetaEnv

        # Consistency Args
        self.cm_alpha = cm_alpha
        self.cm_steps = cm_steps

        # Load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # Set mode for env
        self.env.mode = phase

        # Get data
        batch_task_data, batch_processed_data = batch
        batch_size = len(batch_task_data)

        # Train
        if phase == "train":
            metrics = self._train_step(
                batch_task_data=batch_task_data, 
                batch_processed_data=batch_processed_data, 
            )

        # Val
        elif phase == "val":
            metrics = self._val_step(
                batch_task_data=batch_task_data, 
                batch_processed_data=batch_processed_data, 
            )
            
        # Others
        else:
            raise NotImplementedError()

        # Log
        log_metrics = dict()
        for k, v in metrics.items():
            log_metrics[f"{phase}/{k}"] = v
        for k, v in log_metrics.items():
            self.log(
            k, v, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        
        # return
        return log_metrics["train/loss"] if phase == "train" else log_metrics

    def _train_step(
        self, 
        batch_task_data: List[TaskBase], 
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclass must implement this method")

    def _val_step(
        self, 
        batch_task_data: List[TaskBase], 
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclass must implement this method")
        
    def _solve_step(
        self, 
        batch_task_data: List[TaskBase], 
        batch_processed_data: MetaDataBatch,
        solve_runs: int = 1
    ):
        raise NotImplementedError("Subclass must implement this method")