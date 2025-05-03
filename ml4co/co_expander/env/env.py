import os
import torch
import random
import numpy as np
from ml4co_kit import *
from torch import Tensor
from typing import List, Any, Tuple
from torch.utils.data import DataLoader, Dataset
from co_expander.env.denser import COExpanderDenser
from co_expander.env.sparser import COExpanderSparser


class COExpanderFakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])


class COExpanderEnv(BaseEnv):
    def __init__(
        self,
        # basic
        task: str = None,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        test_batch_size: int = 4,
        num_workers: int = 4,
        sparse_factor: int = 50,
        device: str = "cpu",
        train_folder: str = None,
        val_path: str = None,
        store_data: bool = True,
    ):
        super().__init__(
            name="COExpanderEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
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
        self.atsp_solver = ATSPSolver()
        self.cvrp_solver = CVRPSolver()
        self.mcl_solver = MClSolver()
        self.mcut_solver = MCutSolver()
        self.mis_solver = MISSolver()
        self.mvc_solver = MVCSolver()
        self.tsp_solver = TSPSolver()
        
        # dataset (Fake)
        self.store_data = store_data
        self.train_dataset = COExpanderFakeDataset(train_data_size)
        self.val_dataset = COExpanderFakeDataset(val_data_size)
          
        # data_processor (sparser and denser)
        if self.sparse:
            self.data_processor = COExpanderSparser(self.sparse_factor, self.device)
        else:
            self.data_processor = COExpanderDenser(self.device)
        
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
        if self.task == "ATSP":
            return self.generate_val_data_atsp(begin_idx, end_idx)
        elif self.task == "CVRP":
            return self.generate_val_data_cvrp(begin_idx, end_idx)
        elif self.task == "MCl":
            return self.generate_val_data_mcl(begin_idx, end_idx)
        elif self.task == "MCut":
            return self.generate_val_data_mcut(begin_idx, end_idx)
        elif self.task == "MIS":
            return self.generate_val_data_mis(begin_idx, end_idx)
        elif self.task == "MVC":
            return self.generate_val_data_mvc(begin_idx, end_idx)
        elif self.task == "TSP":
            return self.generate_val_data_tsp(begin_idx, end_idx) 

    def generate_val_data_atsp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.atsp_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "dists": self.atsp_solver.dists,
                "ref_tours": self.atsp_solver.ref_tours
            }
        return self.data_processor.atsp_batch_data_process(
            dists=self.val_data_cache["dists"][begin_idx:end_idx], 
            ref_tours=self.val_data_cache["ref_tours"][begin_idx:end_idx]
        )
    
    def generate_val_data_cvrp(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.cvrp_solver.from_txt(self.val_path, ref=True)
            demands = self.cvrp_solver.demands
            capacities = self.cvrp_solver.capacities
            demands = demands / capacities.reshape(-1, 1)  
            self.val_data_cache = {
                "depots": self.cvrp_solver.depots,
                "points": self.cvrp_solver.points,
                "demands": demands,
                "ref_tours": self.cvrp_solver.ref_tours
            }
              
        return self.data_processor.cvrp_batch_data_process(
            depots=self.val_data_cache["depots"][begin_idx:end_idx],
            points=self.val_data_cache["points"][begin_idx:end_idx], 
            demands=self.val_data_cache["demands"][begin_idx:end_idx],
            ref_tours=self.val_data_cache["ref_tours"][begin_idx:end_idx]
        )

    def generate_val_data_mcl(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcl_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mcl_solver.graph_data
            }
        return self.data_processor.mcl_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )

    def generate_val_data_mcut(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcut_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mcut_solver.graph_data
            }
        return self.data_processor.mcut_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )
    
    def generate_val_data_mis(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mis_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mis_solver.graph_data
            }
        return self.data_processor.mis_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )
    
    def generate_val_data_mvc(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mvc_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mvc_solver.graph_data
            }
        return self.data_processor.mvc_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )

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
        if self.task == "ATSP":
            return self.generate_train_data_atsp(batch_size)
        elif self.task == "CVRP":
            return self.generate_train_data_cvrp(batch_size)
        elif self.task == "MCl":
            return self.generate_train_data_mcl(batch_size)
        elif self.task == "MCut":
            return self.generate_train_data_mcut(batch_size)
        elif self.task == "MIS":
            return self.generate_train_data_mis(batch_size)
        elif self.task == "MVC":
            return self.generate_train_data_mvc(batch_size)
        elif self.task == "TSP":
            return self.generate_train_data_tsp(batch_size) 

    def generate_train_data_atsp(self, batch_size: int)  -> Any:
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
                print(f"\nload atsp train data from {sel_train_sub_file_path}")
                self.atsp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "dists": self.atsp_solver.dists,
                    "ref_tours": self.atsp_solver.ref_tours,
                    "data_size": self.atsp_solver.dists.shape[0]
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        dists = self.train_data_cache["dists"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.atsp_batch_data_process(dists, ref_tours)
    
    def generate_train_data_cvrp(self, batch_size: int) -> Any:
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
                print(f"\nload cvrp train data from {sel_train_sub_file_path}")
                self.cvrp_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                demands = self.cvrp_solver.demands
                capacities = self.cvrp_solver.capacities
                demands = demands / capacities.reshape(-1, 1)
                self.train_data_cache = {
                    "depots": self.cvrp_solver.depots,
                    "points": self.cvrp_solver.points,
                    "demands": demands,
                    "ref_tours": self.cvrp_solver.ref_tours,
                    "data_size": self.cvrp_solver.depots.shape[0]
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        depots = self.train_data_cache["depots"][begin_idx:end_idx]
        points = self.train_data_cache["points"][begin_idx:end_idx]
        demands = self.train_data_cache["demands"][begin_idx:end_idx]
        ref_tours = self.train_data_cache["ref_tours"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.cvrp_batch_data_process(depots, points, demands, ref_tours)

    def generate_train_data_mcl(self, batch_size: int) -> Any:
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
                print(f"\nload mcl train data from {sel_train_sub_file_path}")
                self.mcl_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "graph_data": self.mcl_solver.graph_data,
                    "data_size": len(self.mcl_solver.graph_data)
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        graph_data = self.train_data_cache["graph_data"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mcl_batch_data_process(graph_data)

    def generate_train_data_mcut(self, batch_size: int) -> Any:
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
                print(f"\nload mcut train data from {sel_train_sub_file_path}")
                self.mcut_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "graph_data": self.mcut_solver.graph_data,
                    "data_size": len(self.mcut_solver.graph_data)
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
                
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        graph_data = self.train_data_cache["graph_data"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
            
        # sparse process
        return self.data_processor.mcut_batch_data_process(graph_data)

    def generate_train_data_mis(self, batch_size: int) -> Any:
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
                print(f"\nload mis train data from {sel_train_sub_file_path}")
                self.mis_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "graph_data": self.mis_solver.graph_data,
                    "data_size": len(self.mis_solver.graph_data)
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        graph_data = self.train_data_cache["graph_data"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mis_batch_data_process(graph_data)

    def generate_train_data_mvc(self, batch_size: int) -> Any:
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
                print(f"\nload mvc train data from {sel_train_sub_file_path}")
                self.mvc_solver.from_txt(sel_train_sub_file_path, show_time=True, ref=True)
                self.train_data_cache = {
                    "graph_data": self.mvc_solver.graph_data,
                    "data_size": len(self.mvc_solver.graph_data)
                }
                if self.store_data:
                    self.train_data_historty_cache[sel_train_sub_file_path] = self.train_data_cache
            
            # update cache and index
            self.train_data_cache_idx = 0
            begin_idx = self.train_data_cache_idx
            end_idx = begin_idx + batch_size
        
        # retrieve a portion of data from the cache
        graph_data = self.train_data_cache["graph_data"][begin_idx:end_idx]
        self.train_data_cache_idx = end_idx
        
        # data process
        return self.data_processor.mvc_batch_data_process(graph_data)
     
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

    #################################
    #          Random Mask          #
    #################################

    def random_mask_sparse(
        self, graph_list: List[Tensor], mask: Tensor, 
        ground_truth: Tensor, nodes_num_list: list, 
        edges_num_list: list, prompt_prob: float, delta_scale: tuple
    ):
        if random.uniform(a=0, b=1) < prompt_prob:
            delta = random.uniform(a=delta_scale[0], b=delta_scale[1])
        else:
            return mask

        if self.task == "MCl":
            return self.data_processor.mcl_batch_random_mask(
                graph_list=graph_list, mask=mask, ground_truth=ground_truth, 
                nodes_num_list=nodes_num_list, delta=delta
            )
        elif self.task == "MCut":
            return self.data_processor.mcut_batch_random_mask(
                mask=mask, nodes_num_list=nodes_num_list, delta=delta
            )
        elif self.task == "MIS":
            return self.data_processor.mis_batch_random_mask(
                graph_list=graph_list, mask=mask, ground_truth=ground_truth, 
                nodes_num_list=nodes_num_list, delta=delta
            )
        elif self.task == "MVC":
            return self.data_processor.mvc_batch_random_mask(
                mask=mask, nodes_num_list=nodes_num_list, delta=delta
            )
        elif self.task == "TSP":
            return self.data_processor.tsp_batch_random_mask(
                mask=mask, edges_num_list=edges_num_list, delta=delta
            )
        else:
            raise NotImplementedError()

    def random_mask_dense(
        self, mask: Tensor, raw_data_list: list, prompt_prob: float, delta_scale: tuple
    ):
        if random.uniform(a=0, b=1) < prompt_prob:
            delta = random.uniform(a=delta_scale[0], b=delta_scale[1])
        else:
            return mask
        if self.task == "ATSP":
            return self.data_processor.atsp_batch_random_mask(
                mask=mask, delta=delta, raw_data_list=raw_data_list
            )
        elif self.task == "CVRP":
            return self.data_processor.cvrp_batch_random_mask(
                mask=mask, delta=delta, raw_data_list=raw_data_list
            )
        elif self.task == "TSP":
            return self.data_processor.tsp_batch_random_mask(
                mask=mask, delta=delta
            )

    #################################
    #          Update Mask          #
    #################################
    
    def update_mask_sparse(
        self, heatmap: Tensor, x: Tensor, e: Tensor, graph_list: List[Tensor], 
        mask: Tensor, nodes_num_list: list, edges_num_list: list, 
        max_steps: int, continue_flag: bool, beam_size: int = -1
    ) -> Tuple[Tensor, Tensor]:
        if self.task == "MCl":
            return self.data_processor.mcl_batch_update_mask(
                heatmap=heatmap, x=x, graph_list=graph_list, 
                mask=mask, nodes_num_list=nodes_num_list,
                max_steps=max_steps, continue_flag=continue_flag,
                beam_size=beam_size 
            )
        elif self.task == "MCut":
            return self.data_processor.mcut_batch_update_mask(
                heatmap=heatmap, x=x, graph_list=graph_list, 
                mask=mask, nodes_num_list=nodes_num_list,
                max_steps=max_steps, continue_flag=continue_flag 
            )
        elif self.task == "MIS":
            return self.data_processor.mis_batch_update_mask(
                heatmap=heatmap, x=x, graph_list=graph_list, 
                mask=mask, nodes_num_list=nodes_num_list,
                max_steps=max_steps, continue_flag=continue_flag,
                beam_size=beam_size 
            )
        elif self.task == "MVC":
            return self.data_processor.mvc_batch_update_mask(
                heatmap=heatmap, x=x, graph_list=graph_list, 
                mask=mask, nodes_num_list=nodes_num_list,
                max_steps=max_steps, continue_flag=continue_flag 
            )
        elif self.task == "TSP":
            return self.data_processor.tsp_batch_update_mask(
                heatmap=heatmap, e=e, mask=mask, edges_num_list=edges_num_list,
                max_steps=max_steps, continue_flag=continue_flag
            )
        else:
            raise NotImplementedError()

    def update_mask_dense(
        self, heatmap: Tensor, e: Tensor, raw_data_list: list, 
        mask: Tensor, max_steps: int, continue_flag: bool
    ) -> Tuple[Tensor, Tensor]:
        if self.task == "ATSP":
            return self.data_processor.atsp_batch_update_mask(
                heatmap=heatmap, e=e, mask=mask,
                max_steps=max_steps, continue_flag=continue_flag
            )
        elif self.task == "CVRP":
            return self.data_processor.cvrp_batch_update_mask(
                heatmap=heatmap, e=e, mask=mask, raw_data_list=raw_data_list,
                max_steps=max_steps, continue_flag=continue_flag
            )
        elif self.task == "TSP":
            return self.data_processor.tsp_batch_update_mask(
                heatmap=heatmap, e=e, mask=mask,
                max_steps=max_steps, continue_flag=continue_flag
            )
        else:
            raise NotImplementedError()
            
    #################################
    #            Finetune           #
    #################################
    
    def finetune_sparse(
        self, pred: Tensor, edges_feature, edge_index: Tensor
    ) -> Tensor:
        if self.task == "MCl":
            return self.data_processor.mcl_finetune()
        elif self.task == "MCut":
            return self.data_processor.mcut_finetune(
                pred=pred, edges_feature=edges_feature, edge_index=edge_index
            )
        elif self.task == "MIS":
            return self.data_processor.mis_finetune()
        elif self.task == "MVC":
            return self.data_processor.mvc_finetune() 
        elif self.task == "TSP":
            return self.data_processor.tsp_finetune()
        else:
            raise NotImplementedError()
        
    def finetune_dense(self, pred: Tensor, graph: Tensor) -> Tensor:
        if self.task == "ATSP":
            return self.data_processor.atsp_finetune()
        elif self.task == "CVRP":
            return self.data_processor.cvrp_finetune()
        elif self.task == "TSP":
            return self.data_processor.tsp_finetune()
        else:
            raise NotImplementedError()