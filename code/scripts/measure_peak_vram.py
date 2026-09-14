"""Measure the peak GPU memory a chapter's training script needs on YOUR hardware.

Runs any training module in-process for a few steps and prints per-device peak
memory, so you can check a chapter against your card before committing to a run.
This is how the numbers in ACCELERATORS.md ("GPU requirements at a glance") were
produced on the book's reference A30s.

Examples (run from code/ with the venv active; pick the GPUs with CUDA_VISIBLE_DEVICES):

    CUDA_VISIBLE_DEVICES=0 python -m scripts.measure_peak_vram chapter05.train_lora \
        --train data/it_support_fmt/train.jsonl --valid data/it_support_fmt/valid.jsonl \
        --out /tmp/probe_lora --max_steps 3 --report_to none

    CUDA_VISIBLE_DEVICES=0,1 python -m scripts.measure_peak_vram chapter06.train_sft \
        --train data/it_support_fmt/train.jsonl --valid data/it_support_fmt/valid.jsonl \
        --out /tmp/probe_sft --max_steps 3 --report_to none

Everything after the module name is passed to that module unchanged. An
out-of-memory error is reported as a result, not a crash, with the peak reached
before the failure. Delete the --out directory afterwards (full SFT writes ~8 GB).
"""
from __future__ import annotations

import json
import runpy
import sys
import time
import traceback

import torch

from common.gpu import peak_gpu_memory


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    t0 = time.time()
    status, error = "ok", ""
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit:
        pass
    except BaseException as exc:  # noqa: BLE001 - we want OOM reported as a result
        status = "ERROR"
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        traceback.print_exc()
    result = {
        "module": module,
        "status": status,
        "error": error,
        "seconds": round(time.time() - t0),
        "torch": torch.__version__,
        "devices": peak_gpu_memory(),
    }
    total = sum(float(d["peak_allocated_gib"]) for d in result["devices"])
    print("\n=== Peak GPU memory ===")
    for d in result["devices"]:
        print(f"  cuda:{d['device']} ({d['name']}): {d['peak_allocated_gib']:.2f} GiB allocated, "
              f"{d['peak_reserved_gib']:.2f} GiB reserved")
    if len(result["devices"]) > 1:
        print(f"  total across GPUs: {total:.2f} GiB (what a single card would need)")
    if status != "ok":
        print(f"  run ended with {error}")
    print("PEAK_VRAM_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
