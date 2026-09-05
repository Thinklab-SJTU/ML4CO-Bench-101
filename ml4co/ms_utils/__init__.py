"""Shared MindSpore runtime helpers for ML4CO MS backends."""

from .ms_runtime_env import setup_ld_library_path

# Critical: run before importing anything that pulls native libs.
setup_ld_library_path()

from .skip_ortools import apply as skip_ortools
from .type_utils import to_numpy, to_tensor
from .device_utils import (
    MS_ASCEND,
    MS_CPU,
    MS_GPU,
    current_ms_device,
    ensure_ms_device,
    maybe_move_tensor,
    normalize_ms_device,
    set_ms_device,
    tensor_device_target,
)

__all__ = [
    "setup_ld_library_path",
    "skip_ortools",
    "to_numpy",
    "to_tensor",
    "normalize_ms_device",
    "set_ms_device",
    "ensure_ms_device",
    "maybe_move_tensor",
    "current_ms_device",
    "tensor_device_target",
    "MS_CPU",
    "MS_GPU",
    "MS_ASCEND",
]
