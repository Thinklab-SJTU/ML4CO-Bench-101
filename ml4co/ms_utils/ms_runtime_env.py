"""Process env tweaks that must run before importing MindSpore / native libs."""

from __future__ import annotations

import os


def setup_ld_library_path() -> None:
    """
    Prepend ``$CONDA_PREFIX/lib`` to ``LD_LIBRARY_PATH``.

    Needed on Ascend hosts so conda ``libstdc++`` (GLIBCXX_3.4.32+) is found
    before the system one. Call this before importing mindspore / ml4co_kit.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(lib_dir):
        return
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(":") if p]
    if lib_dir in parts:
        # Keep conda lib first even if it already appears later.
        parts = [p for p in parts if p != lib_dir]
    os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir, *parts])
