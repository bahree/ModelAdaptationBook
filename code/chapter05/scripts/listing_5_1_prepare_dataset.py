"""Step 1 of the hands-on project: download Dolly 15K and prepare a subset.

This script:
  1. Downloads the Databricks Dolly 15K dataset from Hugging Face (first run only;
     subsequent runs use the cached copy).
  2. Filters examples by length (--min_length / --max_length).
  3. Shuffles with a fixed seed and splits into train/valid/test.
  4. Converts each example to messages format (system, user, assistant) and
     writes train.jsonl, valid.jsonl, test.jsonl, and manifest.json to --out.

Run from the repo root (code/) so that chapter05 and common are importable.
Example:
  python chapter05/scripts/listing_5_1_prepare_dataset.py \\
    --out chapter05/data/dolly_subset --seed 42 --train 400 --valid 50 --test 50
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT
from common.jsonl import write_jsonl
from common.manifest import write_json


def dolly_to_messages(
    instruction: str,
    context: str | None,
    response: str,
    *,
    system_prompt: str,
) -> Dict[str, Any]:
    """Convert Dolly format (instruction, context, response) to messages format.
    
    Dolly format:
    - instruction: The task/question
    - context: Optional background information
    - response: The answer/output
    
    We combine instruction + context (if present) into the user message.
    """
    # Combine instruction and context for user message
    if context and context.strip():
        user_content = f"{context}\n\n{instruction}"
    else:
        user_content = instruction
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation.

    Returns:
        Namespace with output path, seed, split sizes, system prompt, and
        length filter thresholds.
    """
    ap = argparse.ArgumentParser(
        description="Prepare a subset of Databricks Dolly 15K for LoRA fine-tuning."
    )
    ap.add_argument("--out", required=True, help="Output folder (will create train/valid/test.jsonl)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--train", type=int, default=400, help="Number of training examples")
    ap.add_argument("--valid", type=int, default=50, help="Number of validation examples")
    ap.add_argument("--test", type=int, default=50, help="Number of test examples")
    ap.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use for all examples",
    )
    ap.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="Minimum character length for instruction+response (filter short examples)",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum character length for instruction+response (filter very long examples)",
    )
    return ap.parse_args()


def main() -> None:
    """Download Dolly 15K, filter by length, split, convert to messages, and write JSONL."""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Download and prepare dataset")
    print("  Downloading Databricks Dolly 15K from Hugging Face (first run may take a minute)...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    print("  Loaded. Filtering and splitting...")
    
    # Filter and shuffle
    import random
    rng = random.Random(args.seed)
    
    filtered_examples = []
    for example in ds:
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        response = example.get("response", "")
        
        # Calculate total length (instruction + context + response)
        total_length = len(instruction) + len(context or "") + len(response)
        
        if args.min_length <= total_length <= args.max_length:
            filtered_examples.append(example)
    
    print(f"Filtered to {len(filtered_examples)} examples (length {args.min_length}-{args.max_length} chars)")
    
    # Shuffle with seed
    rng.shuffle(filtered_examples)
    
    total_needed = args.train + args.valid + args.test
    if len(filtered_examples) < total_needed:
        raise RuntimeError(
            f"Not enough examples after filtering: have {len(filtered_examples)}, need {total_needed}"
        )
    
    # Split into train/valid/test
    train_examples = filtered_examples[: args.train]
    valid_examples = filtered_examples[args.train : args.train + args.valid]
    test_examples = filtered_examples[args.train + args.valid : args.train + args.valid + args.test]
    
    # Convert to messages format, preserving category info
    train_rows = []
    for ex in train_examples:
        msg_row = dolly_to_messages(
            ex["instruction"],
            ex.get("context"),
            ex["response"],
            system_prompt=args.system_prompt,
        )
        # Preserve category for evaluation
        msg_row["category"] = ex.get("category", "unknown")
        train_rows.append(msg_row)
    
    valid_rows = []
    for ex in valid_examples:
        msg_row = dolly_to_messages(
            ex["instruction"],
            ex.get("context"),
            ex["response"],
            system_prompt=args.system_prompt,
        )
        msg_row["category"] = ex.get("category", "unknown")
        valid_rows.append(msg_row)
    
    test_rows = []
    for ex in test_examples:
        msg_row = dolly_to_messages(
            ex["instruction"],
            ex.get("context"),
            ex["response"],
            system_prompt=args.system_prompt,
        )
        msg_row["category"] = ex.get("category", "unknown")
        test_rows.append(msg_row)
    
    # Count categories for manifest
    train_categories = Counter(ex.get("category", "unknown") for ex in train_examples)
    
    # Write JSONL files
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    
    # Write manifest
    manifest = {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "filters": {
            "min_length": args.min_length,
            "max_length": args.max_length,
        },
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "category_distribution": dict(train_categories),
        "system_prompt": args.system_prompt,
    }
    write_json(out_dir / "manifest.json", manifest)
    
    print(f"\n✓ Wrote Dolly 15K subset to: {out_dir}")
    print(f"  - Train: {len(train_rows)} examples")
    print(f"  - Valid: {len(valid_rows)} examples")
    print(f"  - Test: {len(test_rows)} examples")
    print(f"  - Categories: {dict(train_categories)}")


if __name__ == "__main__":
    main()
