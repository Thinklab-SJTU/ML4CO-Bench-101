import os
import torch
import numpy as np
from typing import Any
from ml4co_kit import *
from torch.utils.data import DataLoader, Dataset
from dimes.env.denser import DIMESDenser
from dimes.env.sparser import DIMESSparser


class FakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])


class DIMESEnv(BaseEnv):
    def __init__(
        self,
        task: str = None,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        sparse_factor: int = 50,
        device: str = "cpu",
        train_folder: str = None,
        val_path: str = None,
        store_data: bool = True,
    ):
        super().__init__(
            name="DIMESEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # basic
        self.task = task
        self.sparse = sparse_factor > 0
        self.sparse_factor = sparse_factor
        
        # train data folder and val path
        self.train_folder = train_folder
        self.val_path = val_path
        
        # ml4co-kit solver
        self.tsp_solver = TSPSolver()
        
        # dataset (Fake)
        self.store_data = store_data
        self.train_dataset = FakeDataset(train_data_size)
        self.val_dataset = FakeDataset(val_data_size)
          
        # data_processor (sparser and denser)
        if self.sparse:
            self.data_processor = DIMESSparser(self.sparse_factor, self.device)
        else:
            self.data_processor = DIMESDenser(self.device)
        
        # load data
        if self.mode is not None:
            self.load_data()

    def load_data(self):
        if self.mode == "train":
            self.train_sub_files = [
                os.path.join(self.train_folder, train_files) \
                    for train_files in os.listdir(self.train_folder) 
            ]
            self.train_sub_files_num = len(self.train_sub_files)
            self.train_data_historty_cache = dict()
            self.train_data_cache = None
            self.val_data_cache = None
            self.train_data_cache_idx = 0
        else:
            pass
        
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

    #################################
    #       Generate Val Data       #           
    #################################

    def generate_val_data(self, val_idx: int) -> Any:
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        if self.task == "TSP":
            return self.generate_val_data_tsp(begin_idx, end_idx)
        else:
            raise NotImplementedError()
    
    def generate_val_data_tsp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.tsp_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "points": self.tsp_solver.points,
                "ref_tours": self.tsp_solver.ref_tours
            }
        return self.data_processor.tsp_batch_data_process(
            points=self.val_data_cache["points"][begin_idx:end_idx], 
            ref_tours=self.val_data_cache["ref_tours"][begin_idx:end_idx]
        )   
        
    #################################
    #      Generate Train Data      #
    #################################
    
    def generate_train_data(self, batch_size: int) -> Any:
        if self.task == "TSP":
            return self.generate_train_data_tsp(batch_size)
        else:
            raise NotImplementedError()

    def generate_train_data_tsp(self, batch_size: int) -> Any:
        # check data cache
        begin_idx = self.train_data_cache_idx
        end_idx = begin_idx + batch_size
        if self.train_data_cache is None or end_idx > self.train_data_cache["data_size"]:
            # select one train file randomly
            sel_idx = np.random.randint(low=0, high=self.train_sub_files_num, size=(1,))[0]
            sel_train_sub_file_path = self.train_sub_files[sel_idx]

            # check if the data is in the cache when store_data is True
            if self.store_data and sel_train_sub_file_path in self.train_data_historty_cache.keys():
                # using data cache if the data is in the cache
                print(f"\nusing data cache ({sel_train_sub_file_path})")
                self.train_data_cache = self.train_data_historty_cache[sel_train_sub_file_path]
            else: 
                # load data from the train file
                print(f"\nload tsp train data from {sel_train_sub_file_path}")
                self.tsp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "points": self.tsp_solver.points,
                    "ref_tours": self.tsp_solver.ref_tours,
                    "data_size": self.tsp_solver.points.shape[0]
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
    
        # retrieve a portion of data from the cache
        points = self.train_data_cache["points"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
            
        # data process
        return self.data_processor.tsp_batch_data_process(points, ref_tours)