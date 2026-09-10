"""Peak GPU-memory reporting shared by every training script.

Every training entrypoint calls ``report_peak_gpu_memory()`` right after
``trainer.train()`` so each run prints the number that decides whether the
chapter fits your card. The measured values for the book's reference GPU
(NVIDIA A30, 24 GB) live in ``ACCELERATORS.md`` ("GPU requirements at a
glance"); re-measure any chapter with ``python -m scripts.measure_peak_vram``.
"""
from __future__ import annotations

from typing import Dict, List

import torch

GiB = float(2**30)


def reset_peak_gpu_memory() -> None:
    """Reset the CUDA peak counters (call before training if the model was already loaded)."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_gpu_memory() -> List[Dict[str, object]]:
    """Per-device peak allocated / reserved memory in GiB (empty list without CUDA)."""
    if not torch.cuda.is_available():
        return []
    return [
        {
            "device": i,
            "name": torch.cuda.get_device_name(i),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(i) / GiB, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(i) / GiB, 2),
        }
        for i in range(torch.cuda.device_count())
    ]


def report_peak_gpu_memory(label: str = "training") -> List[Dict[str, object]]:
    """Print and return per-device peaks. With ``device_map="auto"`` on a multi-GPU
    box the model is sharded, so the sum across devices is the number to compare
    against a single card."""
    stats = peak_gpu_memory()
    if not stats:
        return stats
    print(f"\nPeak GPU memory during {label}:")
    for s in stats:
        print(f"  cuda:{s['device']} ({s['name']}): {s['peak_allocated_gib']:.2f} GiB allocated, "
              f"{s['peak_reserved_gib']:.2f} GiB reserved")
    if len(stats) > 1:
        total = sum(float(s["peak_allocated_gib"]) for s in stats)
        print(f"  total across {len(stats)} GPUs: {total:.2f} GiB allocated "
              f"(this is what a single card would need)")
    return stats
