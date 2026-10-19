"""Device placement shared by every chapter.

``device_map="auto"`` is the right choice on a CUDA box, but on Apple Silicon (MPS)
or CPU it can offload layers to CPU/disk. That offload breaks PEFT adapter loading
(``KeyError`` in ``_update_offload``), makes generation extremely slow, and corrupts
LoRA training (NaN gradients, backward device-mismatch errors). Every script that
loads a model goes through ``resolve_device_map`` so those machines get a single
pinned device instead. See LESSONS.md, lesson 1.
"""
from __future__ import annotations

from typing import Any

import torch


def has_mps() -> bool:
    """True on Apple Silicon (Metal/MPS) builds of PyTorch."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device_map(device_map: Any = "auto") -> Any:
    """Resolve "auto" to a concrete placement that avoids CPU/disk offload.

    CUDA keeps "auto" (a single 24 GB+ card fits the 4B model with no offload, and
    the multi-GPU chapters rely on sharding). MPS and CPU are pinned to one device,
    which builds no offload index. Any value other than "auto" is returned unchanged.
    """
    if device_map != "auto":
        return device_map
    if torch.cuda.is_available():
        return "auto"
    if has_mps():
        return {"": "mps"}
    return {"": "cpu"}
