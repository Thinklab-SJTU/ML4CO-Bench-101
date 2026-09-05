"""MindSpore device helpers: keep ``env.device`` and runtime context in sync."""


import mindspore as ms
from typing import Optional, Union


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


def set_ms_device(device: Optional[Union[str, object]], device_id: int = 0) -> str:
    """
    Set process-level MindSpore device to match ``device``.

    Returns the canonical target string (``CPU`` / ``GPU`` / ``Ascend``).
    Must be called before model forward; otherwise tensors follow a stale context.
    """
    target = normalize_ms_device(device)

    # Prefer PYNATIVE for debugging parity with PyTorch.
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE)
    except Exception:
        pass

    # Prefer ``set_device`` (``device_target`` in set_context is deprecated).
    if hasattr(ms, "set_device"):
        try:
            if target == MS_CPU:
                ms.set_device(MS_CPU)
            else:
                ms.set_device(target, int(device_id))
            return target
        except TypeError:
            try:
                ms.set_device(target)
                return target
            except Exception:
                pass
        except Exception:
            pass

    # Fallback for older MindSpore
    ctx = {"mode": ms.PYNATIVE_MODE, "device_target": target}
    if target != MS_CPU:
        ctx["device_id"] = int(device_id)
    try:
        ms.set_context(**ctx)
    except Exception:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target)
    return target


def ensure_ms_device(device: Optional[Union[str, object]], device_id: int = 0) -> str:
    """Idempotent alias of ``set_ms_device`` for call sites before each step."""
    return set_ms_device(device, device_id=device_id)
