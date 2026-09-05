"""Shared MindSpore runtime helpers for ML4CO MS backends."""

from .ms_runtime_env import setup_ld_library_path

# Critical: run before importing anything that pulls native libs (e.g. type_utils -> mindspore).
setup_ld_library_path()

from .skip_ortools import apply as skip_ortools

__all__ = [
    "setup_ld_library_path",
    "skip_ortools",
    "to_numpy",
    "to_tensor",
]


def __getattr__(name: str):
    # Lazy: avoid importing mindspore until to_numpy/to_tensor are needed.
    if name in ("to_numpy", "to_tensor"):
        from .type_utils import to_numpy, to_tensor

        globals()["to_numpy"] = to_numpy
        globals()["to_tensor"] = to_tensor
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
