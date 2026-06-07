"""Monitor SFT training progress from trainer_state.json.

Reads the log history written by HuggingFace Trainer and prints a
summary of training and validation loss at each logged step.

Run from code/:
    python -m chapter06.scripts.monitor chapter06/runs/sft_run1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def monitor(run_dir: str) -> None:
    state_file = Path(run_dir) / "trainer_state.json"
    if not state_file.exists():
        print(f"No trainer_state.json found in {run_dir}")
        print("Training may not have started yet.")
        return

    state = json.loads(state_file.read_text())
    history = state.get("log_history", [])
    if not history:
        print("Log history is empty -- training just started.")
        return

    # Separate training and eval entries
    train_entries = [e for e in history if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in history if "eval_loss" in e]

    print(f"Run directory: {run_dir}")
    print(f"Total logged steps: {len(history)}")
    print(f"Current epoch: {history[-1].get('epoch', 'N/A')}")
    print()

    # Training loss trajectory
    if train_entries:
        first = train_entries[0]
        last = train_entries[-1]
        print("Training loss:")
        print(f"  First logged : {first['loss']:.4f}  (step {first.get('step', '?')})")
        print(f"  Latest       : {last['loss']:.4f}  (step {last.get('step', '?')})")
        delta = last["loss"] - first["loss"]
        print(f"  Change       : {delta:+.4f}")

    # Validation loss trajectory
    if eval_entries:
        best = min(eval_entries, key=lambda e: e["eval_loss"])
        latest = eval_entries[-1]
        print("\nValidation loss:")
        print(f"  Best         : {best['eval_loss']:.4f}  (epoch {best.get('epoch', '?'):.1f})")
        print(f"  Latest       : {latest['eval_loss']:.4f}  (epoch {latest.get('epoch', '?'):.1f})")

        # Warn if overfitting
        if len(eval_entries) >= 2:
            recent = eval_entries[-1]["eval_loss"]
            previous = eval_entries[-2]["eval_loss"]
            if recent > previous + 0.02:
                print("\n  WARNING: Validation loss increased between last two evals.")
                print("  This may indicate overfitting. Consider stopping early.")

    # Gradient norm check
    grad_entries = [e for e in history if "grad_norm" in e]
    if grad_entries:
        norms = [e["grad_norm"] for e in grad_entries]
        max_norm = max(norms)
        print(f"\nGradient norm (max): {max_norm:.2f}")
        if max_norm > 5.0:
            print("  WARNING: Gradient norm exceeded 5.0 -- consider reducing learning rate.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m chapter06.scripts.monitor <run_dir>")
        sys.exit(1)
    monitor(sys.argv[1])
