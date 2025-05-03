import numpy as np
from torch import Tensor
from typing import Union, List, Any
from goal.model.encoder.gat_encoder import GATEncoder
from goal.model.decoder.decode.atsp import atsp_greedy_decode, atsp_beam_decode
from goal.model.decoder.decode.tsp import tsp_greedy_decode, tsp_beam_decode
from goal.model.decoder.decode.cvrp import cvrp_greedy_decode, cvrp_beam_decode
from goal.model.decoder.decode.mis import mis_greedy_decode, mis_beam_decode
from goal.model.decoder.decode.mcl import mcl_greedy_decode, mcl_beam_decode
from goal.model.decoder.decode.mvc import mvc_greedy_decode, mvc_beam_decode


GREEDY_DECODE_FUNC_DICT = {
    "ATSP": atsp_greedy_decode,
    "TSP": tsp_greedy_decode,
    "CVRP": cvrp_greedy_decode,
    "MIS": mis_greedy_decode,
    "MVC": mvc_greedy_decode,
    "MCl": mcl_greedy_decode,
}


BEAM_DECODE_FUNC_DICT = {
    "ATSP": atsp_beam_decode,
    "TSP": tsp_beam_decode,
    "CVRP": cvrp_beam_decode,
    "MIS": mis_beam_decode,
    "MVC": mvc_beam_decode,
    "MCl": mcl_beam_decode,
}


class GOALDecoder(object):
    def __init__(self, knns: int = -1, decode_kwargs: dict = {}) -> None:
        self.knns = knns
        self.decode_kwargs = decode_kwargs

    def decode(
        self, task, x: Tensor, e: Tensor, raw_data: Any, model: GATEncoder, 
        beam_size: int, return_cost: bool
    ) -> Union[List[np.ndarray], np.floating]:
        if beam_size <= 1:  
            decode_func = GREEDY_DECODE_FUNC_DICT[task]
            return decode_func(
                x=x, e=e, raw_data=raw_data, model=model, return_cost=return_cost, 
                knns=self.knns, **self.decode_kwargs
            )
        else:  
            decode_func = BEAM_DECODE_FUNC_DICT[task]
            return decode_func(
                x=x, e=e, raw_data=raw_data, model=model, beam_size=beam_size, 
                return_cost=return_cost, knns=self.knns, **self.decode_kwargs
            )