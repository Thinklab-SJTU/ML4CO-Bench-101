import torch
import numpy as np
from ml4co_kit import *
from torch import Tensor
from typing import Sequence
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset


class FakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])


class UTSPEnv(BaseEnv):
    def __init__(
        self,
        generator: TSPDataGenerator,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        device: str = "cpu",
        val_path: str = None,
        store_data: bool = True,
    ):
        super(UTSPEnv, self).__init__(
            name="UTSPEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )

        
        # ml4co-kit solver and val data
        self.tsp_solver = TSPSolver()
        self.tsp_gen = generator
        self.val_path = val_path
        self.val_data_cache = None
        
        # dataset (Fake)
        self.store_data = store_data
        self.train_dataset = FakeDataset(train_data_size)
        self.val_dataset = FakeDataset(val_data_size)         
        
    def train_dataloader(self):
        train_dataloader=DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader

    def generate_val_data(self, val_idx: int) -> Sequence[Tensor]:
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size

        if self.val_data_cache is None:
            self.tsp_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "points": self.tsp_solver.points
            }
        return self.tsp_batch_data_process(
            points=self.val_data_cache["points"][begin_idx:end_idx]
        )   
    
    def generate_train_data(self, batch_size: int) -> Sequence[Tensor]:
        points = self.tsp_gen.generate_only_instance_for_us(batch_size)
        return self.tsp_batch_data_process(points)
    
    def tsp_batch_data_process(self, points: np.ndarray) -> Sequence[Tensor]:
        # preparation
        x_list = list()
        graph_list = list()
        
        # process core
        for idx in range(points.shape[0]):
            # data process
            x = to_tensor(points[idx])
            graph = to_tensor(cdist(points[idx], points[idx])).float()

            # add to list
            x_list.append(x)
            graph_list.append(graph)
        
        # return
        return (
            torch.stack(x_list, 0).to(self.device),
            torch.stack(graph_list, 0).to(self.device)
        )
        
            
            