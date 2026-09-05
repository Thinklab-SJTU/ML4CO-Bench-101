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
    """Return cached active device target, if any."""
    return None if _ACTIVE is None else _ACTIVE[0]


# Canonical device target of a Tensor (``CPU`` / ``GPU`` / ``Ascend``).
def tensor_device_target(tensor: Tensor) -> str:
    """Canonical device target of a Tensor (``CPU`` / ``GPU`` / ``Ascend``)."""
    return normalize_ms_device(tensor.device)


# Set process-level MindSpore device to match ``device``.
def set_ms_device(
    device: Optional[Union[str, object]],
    device_id: int = 0,
    *,
    force: bool = False,
) -> str:
    """
    Set process-level MindSpore device to match ``device``.

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


def ensure_ms_device(
    device: Optional[Union[str, object]],
    device_id: int = 0,
) -> str:
    """No-op when already on the requested device; otherwise call ``set_ms_device``."""
    return set_ms_device(device, device_id=device_id, force=False)


def maybe_move_tensor(tensor: Optional[Tensor], device: Optional[str]) -> Optional[Tensor]:
    """``move_to`` only when the tensor is not already on ``device``."""
    if tensor is None or device is None:
        return tensor
    target = normalize_ms_device(device)
    try:
        if tensor_device_target(tensor) == target:
            return tensor
        return tensor.move_to(target)
    except Exception:
        return tensor
