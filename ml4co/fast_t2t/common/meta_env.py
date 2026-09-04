from typing import List, Tuple
from torch.utils.data import DataLoader
from ml4co_kit import TaskBase, BaseEnv
from .meta_dataset import MetaDataset, MetaDataBatch


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
        device: str = "cuda",
    ):
        # Super Args
        super(MetaEnv, self).__init__(
            name="MetaEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )

        # Dataset related
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        train_dataloader=DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )
        return test_dataloader

    def process_batch_data(
        self, batch_task_data: List[TaskBase]
    ) -> Tuple[List[TaskBase], MetaDataBatch]:
        raise NotImplementedError("Subclass must implement this method")