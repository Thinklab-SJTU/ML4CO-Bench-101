#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch installed ``ml4co_kit`` so importing the package does not load OR-Tools.

Usage (on the NPU / conda env)::

    conda activate ms_cp311
    python scripts/patch_ml4co_kit_skip_ortools.py

This only edits ``site-packages/ml4co_kit/solver/__init__.py`` (try/except
around the ortools import). It does not change project source.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


OLD = "from .common.ortools import ORSolver"
NEW = """try:
    from .common.ortools import ORSolver
except Exception:  # ImportError / libre2 ABI / protobuf side-effects
    class ORSolver:  # noqa: N801
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ORSolver unavailable (ortools import failed). "
                "Patched by patch_ml4co_kit_skip_ortools.py"
            )
"""


def find_solver_init() -> Path:
    spec = importlib.util.find_spec("ml4co_kit.solver")
    if spec is None or not spec.origin:
        raise RuntimeError("ml4co_kit.solver not found in this environment")
    return Path(spec.origin)


def patch(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if "ORSolver unavailable (ortools import failed)" in text:
        print(f"[skip] already patched: {path}")
        return False
    if OLD not in text:
        raise RuntimeError(
            f"Expected line not found in {path}:\n  {OLD}\n"
            "ml4co_kit layout may have changed; patch manually."
        )
    new_text = text.replace(OLD, NEW, 1)
    if dry_run:
        print(f"[dry-run] would patch: {path}")
        return True
    backup = path.with_suffix(path.suffix + ".bak_ortools")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"[backup] {backup}")
    path.write_text(new_text, encoding="utf-8")
    print(f"[patched] {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = find_solver_init()
    print(f"[target] {path}")
    patch(path, dry_run=args.dry_run)
    print("[hint] verify: python -c \"from ml4co_kit import TSPWrapper; print('OK')\"")


if __name__ == "__main__":
    main()
