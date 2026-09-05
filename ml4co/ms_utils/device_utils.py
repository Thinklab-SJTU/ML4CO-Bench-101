"""MindSpore device helpers: keep ``env.device`` and runtime context in sync."""


import mindspore as ms
from mindspore import Tensor
from typing import Optional, Tuple, Union


# Canonical MindSpore device strings
MS_CPU = "CPU"
MS_GPU = "GPU"
MS_ASCEND = "Ascend"

_ALIAS = {
    "cpu": MS_CPU,
    "gpu": MS_GPU,
    "cuda": MS_GPU,
    "ascend": MS_ASCEND,
    "npu": MS_ASCEND,
}


# Process-level cache: avoid repeated ``ms.set_device`` (very expensive on Ascend).
_ACTIVE: Optional[Tuple[str, int]] = None


# Canonical MindSpore device strings
def normalize_ms_device(device: Optional[Union[str, object]]) -> str:
    """Map user / PyTorch-style names to MindSpore device targets."""
    if device is None:
        return MS_CPU
    raw = str(device).strip()
    # Device objects may print as ``CPU`` / ``Ascend:0``
    key = raw.split(":", 1)[0].lower()
    if key in _ALIAS:
        return _ALIAS[key]
    if raw in (MS_CPU, MS_GPU, MS_ASCEND):
        return raw
    raise ValueError(
        f"Unsupported MindSpore device {device!r}. "
        f"Use one of: CPU, GPU, Ascend (aliases: cpu, cuda/gpu, npu)."
    )


# Return cached active device target, if any.
def current_ms_device() -> Optional[str]:
    return None if _ACTIVE is None else _ACTIVE[0]


# Canonical device target of a Tensor (``CPU`` / ``GPU`` / ``Ascend``).
def tensor_device_target(tensor: Tensor) -> str:
    return normalize_ms_device(tensor.device)


# Set process-level MindSpore device to match ``device``.
def set_ms_device(
    device: Optional[Union[str, object]],
    device_id: int = 0,
    *,
    force: bool = False,
) -> str:
    """
    Idempotent: if the process is already on ``(target, device_id)``, this is a
    no-op. Repeated ``ms.set_device`` / ``set_context`` on Ascend is extremely
    costly and was causing ~10x slowdowns when called every train/solve step.
    """
    global _ACTIVE
    target = normalize_ms_device(device)
    device_id = 0 if target == MS_CPU else int(device_id)
    key = (target, device_id)

    if not force and _ACTIVE == key:
        return target

    # Prefer ``set_device`` (``device_target`` in set_context is deprecated).
    # Only touch context mode once when actually switching devices.
    if hasattr(ms, "set_device"):
        try:
            if _ACTIVE is None:
                try:
                    ms.set_context(mode=ms.PYNATIVE_MODE)
                except Exception:
                    pass
            if target == MS_CPU:
                ms.set_device(MS_CPU)
            else:
                ms.set_device(target, device_id)
            _ACTIVE = key
            return target
        except TypeError:
            try:
                ms.set_device(target)
                _ACTIVE = key
                return target
            except Exception:
                pass
        except Exception:
            pass

    # Fallback for older MindSpore
    ctx = {"mode": ms.PYNATIVE_MODE, "device_target": target}
    if target != MS_CPU:
        ctx["device_id"] = device_id
    try:
        ms.set_context(**ctx)
    except Exception:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target)
    _ACTIVE = key
    return target


# No-op when already on the requested device; otherwise call ``set_ms_device``.
def ensure_ms_device(
    device: Optional[Union[str, object]],
    device_id: int = 0,
) -> str:
    return set_ms_device(device, device_id=device_id, force=False)


# No-op when already on the requested device; otherwise call ``move_to``.
def maybe_move_tensor(
    tensor: Optional[Tensor],
    device: Optional[str],
    *,
    strict: bool = False,
) -> Optional[Tensor]:
    """
    Move ``tensor`` onto ``device`` only when needed.

    Falls back to ``Tensor(asnumpy(), ...)`` after ``ensure_ms_device`` when
    ``move_to`` fails or leaves storage on the wrong device (common after
    CPU checkpoint load / host-side batch construction).
    """
    if tensor is None or device is None:
        return tensor
    target = normalize_ms_device(device)
    try:
        if tensor_device_target(tensor) == target:
            return tensor
    except Exception:
        pass

    # Prefer native move_to when it works.
    try:
        moved = tensor.move_to(target)
        if tensor_device_target(moved) == target:
            return moved
    except Exception:
        moved = None

    # Recreate on the active process device (most reliable on Ascend).
    try:
        ensure_ms_device(target)
        recreated = ms.Tensor(tensor.asnumpy(), dtype=tensor.dtype)
        if tensor_device_target(recreated) == target:
            return recreated
        recreated = recreated.move_to(target)
        if tensor_device_target(recreated) == target:
            return recreated
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Failed to move tensor to {target}: {exc}"
            ) from exc
        return tensor if moved is None else moved

    if strict:
        raise RuntimeError(
            f"Tensor still on {getattr(recreated, 'device', '?')} after "
            f"move to {target}"
        )
    return recreated


def move_net_to_device(
    net,
    device: Optional[Union[str, object]],
    device_id: int = 0,
    *,
    strict: bool = True,
) -> str:
    """
    Move all ``Parameter``s of a ``nn.Cell`` onto ``device``.

    Prefer recreating parameter data with ``ms.Tensor(...)`` **after**
    ``ensure_ms_device``, so storage is allocated on the active device.
    ``Parameter.move_to`` + ``set_data`` alone is unreliable on some Ascend builds.
    """
    from mindspore import nn as ms_nn

    target = ensure_ms_device(device, device_id=device_id)
    if not isinstance(net, ms_nn.Cell):
        return target

    failed = []
    for name, param in net.parameters_and_names():
        try:
            if tensor_device_target(param) == target:
                continue
            # Allocate on the *current* process device (set above).
            new_data = ms.Tensor(param.asnumpy(), dtype=param.dtype)
            param.set_data(new_data)
            if tensor_device_target(param) != target:
                param.set_data(param.data.move_to(target))
            if tensor_device_target(param) != target:
                failed.append(f"{name}-> {param.device}")
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    if failed and strict:
        raise RuntimeError(
            "Failed to move parameters to "
            f"{target} (device_id={device_id}): {failed[:8]}"
            + (" ..." if len(failed) > 8 else "")
        )
    return target


def summarize_net_devices(net) -> dict:
    """Count parameters by device target (for debugging placement)."""
    from collections import Counter
    from mindspore import nn as ms_nn

    if not isinstance(net, ms_nn.Cell):
        return {}
    counts = Counter()
    for param in net.get_parameters():
        try:
            counts[tensor_device_target(param)] += 1
        except Exception:
            counts["unknown"] += 1
    return dict(counts)
