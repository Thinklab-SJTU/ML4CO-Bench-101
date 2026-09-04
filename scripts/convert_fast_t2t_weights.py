#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Lightning Fast-T2T ``.ckpt`` weights to:
  1) plain PyTorch ``.pt`` state_dict under ``weights/fast_t2t/``
  2) MindSpore ``.ckpt`` under ``weights/fast_t2t_ms/``

Usage (from repo root, env with torch + mindspore)::

    conda activate cp310_base
    PYTHONPATH=. python scripts/convert_fast_t2t_weights.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_pt_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """Pull the model ``state_dict`` out of a Lightning checkpoint (or pass-through)."""
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        # Already a plain state_dict
        return ckpt_obj
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")


def pt_name_to_ms_name(pt_name: str) -> str:
    """
    Map PyTorch LayerNorm / GroupNorm affine names to MindSpore.

    PT: ``*.weight`` / ``*.bias``  ->  MS: ``*.gamma`` / ``*.beta``
    for norm layers; Linear / Conv keep weight & bias.
    """
    # LayerNorm under GCN / per_layer_out / GroupNorm under out.0
    if pt_name.endswith(".norm_x.weight") or pt_name.endswith(".norm_e.weight"):
        return pt_name[: -len(".weight")] + ".gamma"
    if pt_name.endswith(".norm_x.bias") or pt_name.endswith(".norm_e.bias"):
        return pt_name[: -len(".bias")] + ".beta"
    # per_layer_out.{i}.0 is LayerNorm
    if ".per_layer_out." in pt_name and pt_name.endswith(".0.weight"):
        return pt_name[: -len(".weight")] + ".gamma"
    if ".per_layer_out." in pt_name and pt_name.endswith(".0.bias"):
        return pt_name[: -len(".bias")] + ".beta"
    # out.0 is GroupNorm32
    if pt_name.endswith("model.out.0.weight") or pt_name.endswith(".out.0.weight"):
        return pt_name[: -len(".weight")] + ".gamma"
    if pt_name.endswith("model.out.0.bias") or pt_name.endswith(".out.0.bias"):
        return pt_name[: -len(".bias")] + ".beta"
    return pt_name


def convert_pt_state_to_ms_param_dict(
    pt_state: Dict[str, torch.Tensor],
    expected_ms_names: List[str] | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Build a MindSpore ``save_checkpoint``-compatible list of (name, numpy array).

    If ``expected_ms_names`` is given, verify coverage and raise on missing keys.
    """
    ms_dict: Dict[str, np.ndarray] = {}
    for pt_name, tensor in pt_state.items():
        ms_name = pt_name_to_ms_name(pt_name)
        arr = tensor.detach().cpu().numpy()
        ms_dict[ms_name] = arr

    if expected_ms_names is not None:
        expected = set(expected_ms_names)
        got = set(ms_dict.keys())
        missing = sorted(expected - got)
        unexpected = sorted(got - expected)
        if missing or unexpected:
            msg = []
            if missing:
                msg.append(f"missing ({len(missing)}): {missing[:8]}...")
            if unexpected:
                msg.append(f"unexpected ({len(unexpected)}): {unexpected[:8]}...")
            raise RuntimeError("PT->MS name mismatch: " + " | ".join(msg))

    return [(name, ms_dict[name]) for name in sorted(ms_dict.keys())]


def get_ms_param_names() -> List[str]:
    """Instantiate an empty MS Fast-T2T model to read expected parameter names."""
    import mindspore as ms
    from ml4co.fast_t2t_ms import TSPModel, TSPEnv, TSPPLModel

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    env = TSPEnv(mode="solve", device="CPU")
    model = TSPModel(hidden_dim=256, num_layers=12)
    pl = TSPPLModel(env=env, model=model)
    return [p.name for p in pl.get_parameters()]


def save_ms_ckpt(param_list: List[Tuple[str, np.ndarray]], out_path: Path) -> None:
    import mindspore as ms
    from mindspore import Tensor

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    # MindSpore expects list[{"name": str, "data": Tensor}]
    ckpt_list = [{"name": name, "data": Tensor(arr)} for name, arr in param_list]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(ckpt_list, str(out_path))


def convert_one(
    ckpt_path: Path,
    pt_out: Path,
    ms_out: Path,
    expected_ms_names: List[str] | None,
) -> None:
    print(f"[load] {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_state = extract_pt_state_dict(ckpt_obj)

    # 1) PyTorch .pt (plain state_dict)
    pt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_state, pt_out)
    print(f"[save] PyTorch state_dict -> {pt_out}  ({len(pt_state)} tensors)")

    # 2) MindSpore .ckpt
    param_list = convert_pt_state_to_ms_param_dict(pt_state, expected_ms_names)
    save_ms_ckpt(param_list, ms_out)
    print(f"[save] MindSpore ckpt     -> {ms_out}  ({len(param_list)} tensors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=ROOT / "weights" / "fast_t2t",
        help="Directory with Lightning .ckpt files",
    )
    parser.add_argument(
        "--pt-dir",
        type=Path,
        default=ROOT / "weights" / "fast_t2t",
        help="Output directory for .pt files (same folder by default)",
    )
    parser.add_argument(
        "--ms-dir",
        type=Path,
        default=ROOT / "weights" / "fast_t2t_ms",
        help="Output directory for MindSpore .ckpt files",
    )
    parser.add_argument(
        "--skip-ms-check",
        action="store_true",
        help="Do not instantiate MS model to verify parameter names",
    )
    args = parser.parse_args()

    ckpts = sorted(args.src_dir.glob("*.ckpt"))
    # Ignore already-converted MS files if someone put them in src
    ckpts = [p for p in ckpts if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {args.src_dir}")

    expected_ms_names = None
    if not args.skip_ms_check:
        print("[info] Building MS model to verify parameter name mapping...")
        expected_ms_names = get_ms_param_names()
        print(f"[info] MS expects {len(expected_ms_names)} parameters")

    for ckpt_path in ckpts:
        stem = ckpt_path.stem  # e.g. tsp100_fast_t2t
        pt_out = args.pt_dir / f"{stem}.pt"
        ms_out = args.ms_dir / f"{stem}.ckpt"
        convert_one(ckpt_path, pt_out, ms_out, expected_ms_names)

    print("[done]")


if __name__ == "__main__":
    main()
