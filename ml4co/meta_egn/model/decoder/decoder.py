import numpy as np
from torch import Tensor
from typing import Union, List
from meta_egn.model.decoder.decode.mcl import mcl_decode_greedy
from meta_egn.model.decoder.decode.mcut import mcut_decode_greedy
from meta_egn.model.decoder.decode.mis import mis_decode_greedy
from meta_egn.model.decoder.decode.mvc import mvc_decode_greedy


DECODE_FUNC_DICT = {
    "MCl": mcl_decode_greedy,
    "MCut": mcut_decode_greedy,
    "MIS": mis_decode_greedy,
    "MVC": mvc_decode_greedy,
}


class MetaEGNDecoder(object):
    def __init__(self, decode_kwargs: dict = {}) -> None:
        self.decode_kwargs = decode_kwargs
        self.ori_rlsa = decode_kwargs["use_rlsa"]

    def decode(
        self, heatmap: Tensor, task: str, edge_index: Tensor, nodes_num: int, 
        return_cost: bool = False, egn_rlsa_flag: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        self.decode_func = DECODE_FUNC_DICT[task]
        self.decode_kwargs["use_rlsa"] = egn_rlsa_flag and self.ori_rlsa
        return self.decode_func(
            heatmap=heatmap,
            num_nodes=nodes_num,
            edge_index=edge_index,
            **self.decode_kwargs
        )