from typing import List, Tuple
from ml4co_kit import TaskBase, BaseEnv
from ml4co_kit.learning.extra_backends.mindspore import MSDataLoader
from .meta_dataset import MetaDataset, MetaDataBatch
from .device_utils import normalize_ms_device, set_ms_device


class MetaEnv(BaseEnv):
    def __init__(
        self,
        mode: str = None,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        train_dataset: MetaDataset = None,
        val_dataset: MetaDataset = None,
        test_dataset: MetaDataset = None,
        num_workers: int = 4,
        device: str = "Ascend", # ['Ascend', 'GPU', 'CPU']
        device_id: int = 0,
    ):
        # Align MindSpore process device with env.device (CPU / GPU / Ascend).
        device = normalize_ms_device(device)
        self.device_id = int(device_id)
        set_ms_device(device, device_id=self.device_id)

        # Super Args
        super(MetaEnv, self).__init__(
            name="MetaEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device,
        )

        # Dataset related
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        # MindSpore DataLoader (single-process; num_workers kept for API parity)
        return MSDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return MSDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return MSDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
        )

    def process_batch_data(
        self, batch_task_data: List[TaskBase]
    ) -> Tuple[List[TaskBase], MetaDataBatch]:
        raise NotImplementedError("Subclass must implement this method")
