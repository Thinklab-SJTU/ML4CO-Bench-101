from .base import U2DCOOutLayer
from .node import NodeOutLayer


OUTLAYER_DICT = {
    "MCl": NodeOutLayer,
    "MIS": NodeOutLayer,
    "MCut": NodeOutLayer,
    "MVC": NodeOutLayer,
}

def get_out_layer_by_task(task: str):
    return OUTLAYER_DICT[task]