"""
Skip loading OR-Tools when importing ``ml4co_kit``.

Ascend / MindIE injects an incompatible ``libre2`` into ``LD_LIBRARY_PATH``,
and recent ``ortools`` wheels also pull ``protobuf>=4/5``, which breaks
MindSpore's generated ``*_pb2.py``. Fast-T2T does not need OR-Tools.

Call ``apply()`` **before** any ``import ml4co_kit`` / ``import ml4co.fast_t2t_ms``.
"""

from __future__ import annotations

import sys
import types
from typing import Any


_APPLIED = False
_MODULE_NAME = "ml4co_kit.solver.common.ortools"


def apply() -> None:
    """Pre-register a stub so ``ml4co_kit.solver`` never imports real ortools."""
    global _APPLIED
    if _APPLIED or _MODULE_NAME in sys.modules:
        _APPLIED = True
        return

    mod = types.ModuleType(_MODULE_NAME)

    class ORSolver:  # noqa: N801 - match kit export name
        """Placeholder; Fast-T2T never constructs this."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ORSolver is disabled (ml4co.ms_utils.skip_ortools). "
                "Install/fix ortools only if you need OR-Tools solvers."
            )

    mod.ORSolver = ORSolver
    sys.modules[_MODULE_NAME] = mod
    _APPLIED = True
