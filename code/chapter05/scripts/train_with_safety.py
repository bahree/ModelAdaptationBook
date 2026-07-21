#!/usr/bin/env python3
"""
Train LoRA adapter with mixed safety data to prevent safety regression.

This script:
1. Loads your task training data (the IT support examples)
2. Loads safety examples (refusal responses)
3. Mixes them (e.g., 80% task, 20% safety)
4. Trains LoRA adapter on the mixed dataset
5. Preserves safety while improving task performance

Usage:
    python chapter05/scripts/train_with_safety.py \
        --task_data data/it_support_fmt/train.jsonl \
        --safety_data chapter05/data/safety_examples.jsonl \
        --safety_ratio 0.2 \
        --out chapter05/runs/it_lora_with_safety
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mix_datasets(
    task_examples: List[Dict],
    safety_examples: List[Dict],
    safety_ratio: float = 0.2,
    seed: int = 42,
) -> List[Dict]:
    """
    Mix task and safety examples.
    
    Args:
        task_examples: Main task training data
        safety_examples: Safety refusal examples
        safety_ratio: Fraction of safety examples (e.g., 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        Mixed dataset with safety examples interspersed
    """
    random.seed(seed)
    
    # Calculate how many of each type
    total = len(task_examples)
    n_task = int(total * (1 - safety_ratio))
    n_safety = int(total * safety_ratio)
    
    print("Dataset composition:")
    print(f"  Task examples: {n_task} ({(1-safety_ratio)*100:.0f}%)")
    print(f"  Safety examples: {n_safety} ({safety_ratio*100:.0f}%)")
    print(f"  Total: {n_task + n_safety}")
    
    # Sample and shuffle
    task_sample = random.sample(task_examples, min(n_task, len(task_examples)))
    
    # Cycle through safety examples if we need more than available
    safety_sample = []
    while len(safety_sample) < n_safety:
        safety_sample.extend(safety_examples)
    safety_sample = safety_sample[:n_safety]
    random.shuffle(safety_sample)
    
    # Combine and shuffle
    mixed = task_sample + safety_sample
    random.shuffle(mixed)
    
    return mixed


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA with mixed safety data"
    )
    parser.add_argument(
        "--task_data",
        type=str,
        required=True,
        help="Path to task training data (e.g., the IT support examples)"
    )
    parser.add_argument(
        "--safety_data",
        type=str,
        required=True,
        help="Path to safety examples (refusal responses)"
    )
    parser.add_argument(
        "--safety_ratio",
        type=float,
        default=0.2,
        help="Fraction of safety examples (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for mixed training data and model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading task data from: {args.task_data}")
    task_examples = load_jsonl(Path(args.task_data))
    print(f"  Loaded {len(task_examples)} task examples")
    
    print(f"\nLoading safety data from: {args.safety_data}")
    safety_examples = load_jsonl(Path(args.safety_data))
    print(f"  Loaded {len(safety_examples)} safety examples")
    
    # Mix datasets
    print(f"\nMixing datasets (safety ratio: {args.safety_ratio})...")
    mixed = mix_datasets(task_examples, safety_examples, args.safety_ratio, args.seed)
    
    # Save mixed dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mixed_path = out_dir / "train_mixed.jsonl"
    with mixed_path.open("w", encoding="utf-8") as f:
        for ex in mixed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved mixed dataset to: {mixed_path}")
    print("\nNext: Train with this mixed dataset using train_lora.py")
    print("\nCommand:")
    print("  python -m chapter05.train_lora \\")
    print(f"    --train {mixed_path} \\")
    print("    --valid data/it_support/valid.jsonl \\")
    print(f"    --out {out_dir / 'adapter'} \\")
    print("    --epochs 3")


if __name__ == "__main__":
    main()
