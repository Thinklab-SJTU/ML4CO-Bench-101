from typing import Any, Dict, List

from mindspore import nn
from ml4co_kit import TaskBase
from ml4co_kit.learning.extra_backends.mindspore import MSBaseModel

from .meta_env import MetaEnv
from .meta_dataset import MetaDataBatch
from ml4co.ms_utils import ensure_ms_device


class MetaPLModel(MSBaseModel):
    """
    MindSpore counterpart of ``ml4co.fast_t2t.common.MetaPLModel``.

    Uses ``MSBaseModel`` + ``MSTrainer`` instead of PyTorch Lightning.
    """

    def __init__(
        self,
        env: MetaEnv,
        model: nn.Cell,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None,
        cm_alpha: float = 0.2,
        cm_steps: int = 5,
    ):
        # Bind MindSpore runtime to env.device before any Cell / Tensor work.
        device_id = getattr(env, "device_id", 0)
        ensure_ms_device(env.device, device_id=device_id)

        # Super Args
        super().__init__(
            env=env,
            model=model,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Hint
        self.env: MetaEnv

        # Consistency Args
        self.cm_alpha = cm_alpha
        self.cm_steps = cm_steps

        # Load pretrained weights if needed
        if weight_path is not None:
            self.load_weights(weight_path)

    def _sync_device(self):
        """Cheap no-op when already on ``env.device`` (cached in ms_utils)."""
        device_id = getattr(self.env, "device_id", 0)
        ensure_ms_device(self.env.device, device_id=device_id)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # Set mode for env
        self.env.mode = phase
        self._sync_device()

        # Get data
        batch_task_data, batch_processed_data = batch

        # Train
        if phase == "train":
            metrics = self._train_step(
                batch_task_data=batch_task_data,
                batch_processed_data=batch_processed_data,
            )
            # MSTrainer expects a loss Tensor or dict containing ``loss``.
            return metrics["loss"]

        # Val
        if phase == "val":
            metrics = self._val_step(
                batch_task_data=batch_task_data,
                batch_processed_data=batch_processed_data,
            )
            return metrics

        # Others
        raise NotImplementedError()

    def _train_step(
        self,
        batch_task_data: List[TaskBase],
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")

    def _val_step(
        self,
        batch_task_data: List[TaskBase],
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement this method")

    def _solve_step(
        self,
        batch_task_data: List[TaskBase],
        batch_processed_data: MetaDataBatch,
        solve_runs: int = 1,
    ):
        raise NotImplementedError("Subclass must implement this method")
