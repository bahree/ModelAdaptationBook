"""Pre-flight validation for Chapter 5 setup.

Checks that the required data files exist, that PyTorch is installed and CUDA
is available, and optionally runs a 2-step LoRA smoke training to verify the
full pipeline works end-to-end before committing to a long training run.

Usage:
    python chapter05/scripts/validate_chapter05.py

Set BOOKCODE_SKIP_MODEL_DOWNLOAD=1 to skip the smoke training step (useful
in CI or environments without GPU).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # chapter05/


def main() -> None:
    """Run all validation checks: data files, PyTorch, CUDA, and smoke training."""
    print("Chapter 5 validation")
    print(f"- Python: {sys.version.split()[0]}")

    # Dataset presence checks (no ML dependencies required)
    required_files = [
        ROOT / "data" / "smoke" / "train.jsonl",
        ROOT / "data" / "smoke" / "valid.jsonl",
        ROOT / "data" / "golden" / "toy_test.jsonl",
        ROOT / "data" / "golden" / "safety_regression_prompts.jsonl",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))
    print("- Datasets: OK")

    # Torch availability
    try:
        import torch

        print(f"- Torch: {torch.__version__}")
        print(f"- CUDA available: {torch.cuda.is_available()}")
    except Exception:
        print("- Torch: not available yet")
        print("  Install PyTorch first, then re-run this script.")
        return

    if os.getenv("BOOKCODE_SKIP_MODEL_DOWNLOAD") == "1":
        print("- Skipping model download/inference (BOOKCODE_SKIP_MODEL_DOWNLOAD=1).")
        return

    # Tiny LoRA smoke training (GPU recommended).
    if not torch.cuda.is_available():
        print("- No CUDA GPU detected; skipping training smoke test (would be too slow on CPU).")
        return

    out_dir = ROOT / "runs" / "validate_lora_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "chapter05.train_lora",
        "--train",
        str(ROOT / "data" / "smoke" / "train.jsonl"),
        "--valid",
        str(ROOT / "data" / "smoke" / "valid.jsonl"),
        "--out",
        str(out_dir),
        "--max_steps",
        "2",
        "--batch_size",
        "1",
        "--grad_accum",
        "1",
        "--max_length",
        "256",
    ]
    print("- Running tiny LoRA smoke training...")
    print("  " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT.parent))  # run from code/

    adapter_cfg = out_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise SystemExit(f"Smoke training did not produce adapter_config.json in {out_dir}")

    print(f"- Smoke training: OK (adapter written to {out_dir})")


if __name__ == "__main__":
    main()

