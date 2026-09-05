"""Process env tweaks that must run before importing MindSpore / native libs."""

from __future__ import annotations

import os
import sys
from typing import List, Optional


_REEXEC_FLAG = "_ML4CO_CONDA_LIB_LD_READY"


def _conda_lib_dir() -> Optional[str]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    lib_dir = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(lib_dir):
        return None
    return lib_dir


def _path_is_same(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normpath(a) == os.path.normpath(b)


def _ld_path_parts() -> List[str]:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    return [p for p in old.split(":") if p]


def setup_ld_library_path(*, reexec: bool = True) -> None:
    """
    Prepend ``$CONDA_PREFIX/lib`` to ``LD_LIBRARY_PATH``, matching::

        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

    Merely assigning ``os.environ`` is **not** enough: the dynamic linker
    reads ``LD_LIBRARY_PATH`` at process start. If conda lib was not already
    first, this function re-execs the current Python process once so that
    subsequent ``dlopen`` (MindSpore, pybind, etc.) sees the new path.
    """
    lib_dir = _conda_lib_dir()
    if lib_dir is None:
        return

    parts = _ld_path_parts()
    already_first = bool(parts) and _path_is_same(parts[0], lib_dir)

    new_parts = [lib_dir] + [p for p in parts if not _path_is_same(p, lib_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts)

    if already_first or not reexec:
        return
    if os.environ.get(_REEXEC_FLAG) == "1":
        return

    # Re-exec so ld.so picks up LD_LIBRARY_PATH like a shell ``export``.
    os.environ[_REEXEC_FLAG] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)
