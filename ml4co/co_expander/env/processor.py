from typing import Any
from torch import Tensor


class DataProcessor(object):
    def __init__(self, device: str) -> None:
        self.device = device
    
    #################################
    #        Raw Data Process       #
    #################################
    
    def atsp_batch_data_process() -> Any:
        raise NotImplementedError()
           
    def cvrp_batch_data_process() -> Any:
        raise NotImplementedError()
    
    def mcl_batch_data_process() -> Any:
        raise NotImplementedError()
    
    def mcut_batch_data_process() -> Any:
        raise NotImplementedError()

    def mis_batch_data_process() -> Any:
        raise NotImplementedError()

    def mvc_batch_data_process() -> Any:
        raise NotImplementedError()
    
    def tsp_batch_data_process() -> Any:
        raise NotImplementedError()
    
    #################################
    #          Random Mask          #
    #################################
        
    def atsp_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def cvrp_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def mcl_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def mcut_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def mis_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def mvc_batch_random_mask() -> Tensor:
        raise NotImplementedError()
    
    def tsp_batch_random_mask() -> Tensor:
        raise NotImplementedError()

    #################################
    #          Update Mask          #
    #################################

    def atsp_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    def cvrp_batch_update_mask() -> Any:
        raise NotImplementedError()
    
    def mcl_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    def mcut_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    def mis_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    def mvc_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    def tsp_batch_update_mask() -> Tensor:
        raise NotImplementedError()
    
    #################################
    #            Finetune           #
    #################################
    
    def atsp_finetune() -> Any:
        raise NotImplementedError()
    
    def cvrp_finetune() -> Any:
        raise NotImplementedError()
    
    def mcl_finetune() -> Any:
        raise NotImplementedError()
  
    def mcut_finetune() -> Any:
        raise NotImplementedError()
    
    def mis_finetune() -> Any:
        raise NotImplementedError()
    
    def mvc_finetune() -> Any:
        raise NotImplementedError()
    
    def tsp_finetune() -> Any:
        raise NotImplementedError()