import os
import torch
import numpy as np
from ml4co_kit import *
from typing import Any
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from ml4co.meta_egn.env.data_processor import MetaEGNDataset


class MetaEGNEnv(BaseEnv):
    def __init__(
        self,
        # basic
        task: str = None,
        mode: str = None,
        train_data_size: int = 12800,
        val_data_size: int = 500,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 4,
        device: str = "cpu",
        train_path: str = None,
        val_path: str = None,
        processed_save_path: str = None
    ):
        super().__init__(
            name="MetaEGNEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # basic
        self.task = task
        
        # train data folder and val path
        self.train_path = train_path
        self.val_path = val_path
        self.processed_save_path = processed_save_path
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        
        # ml4co-kit solver
        self.ml4co_kit_solver_dict = {
            "MCl": MClSolver(),
            "MCut": MCutSolver(),
            "MIS": MISSolver(),
            "MVC": MVCSolver()
        }
        
        # load data
        if self.mode is not None:
            self.load_data()

    def load_data(self):
        if self.mode == "train":
            self.val_dataset = MetaEGNDataset(
                ml4co_kit_solver=self.ml4co_kit_solver_dict[self.task],
                txt_data_path=self.val_path,
                pt_store_path=os.path.join(self.processed_save_path, "val"),
                data_size=self.val_data_size,
                mode="val"
            )
            self.train_dataset = MetaEGNDataset(
                ml4co_kit_solver=self.ml4co_kit_solver_dict[self.task],
                txt_data_path=self.train_path,
                pt_store_path=os.path.join(self.processed_save_path, "train"),
                data_size=self.train_data_size,
                mode="train"
            )
        else:
            pass
        
    def train_dataloader(self):
        train_dataloader=DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False,
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader
