"""Prepare distillation training data: filter, verify, and split.

Takes teacher-generated outputs, applies quality filters, and creates
train/valid splits with a manifest for reproducibility.

Run from code/:
    python -m chapter07.scripts.prepare_distillation_data \
        --input chapter07/data/teacher_outputs.jsonl \
        --out chapter07/data/distill_ready \
        --train 160 --valid 40
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
from collections import Counter
from pathlib import Path

from common.jsonl import read_jsonl, write_jsonl
from common.manifest import write_json
from common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare distillation data")
    ap.add_argument("--input", required=True, help="Teacher output JSONL")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--train", type=int, default=160, help="Training examples")
    ap.add_argument("--valid", type=int, default=40, help="Validation examples")
    ap.add_argument("--min_response_words", type=int, default=10,
                    help="Minimum word count for teacher response")
    ap.add_argument("--max_response_words", type=int, default=500,
                    help="Maximum word count for teacher response")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def quality_filter(example: dict, min_words: int, max_words: int) -> bool:
    """Basic quality filter on teacher responses."""
    msgs = example["messages"]
    assistant_msg = next(
        (m["content"] for m in msgs if m["role"] == "assistant"), ""
    )
    word_count = len(assistant_msg.split())

    # Too short = likely empty or refusal
    if word_count < min_words:
        return False
    # Too long = likely degenerate repetition
    if word_count > max_words:
        return False
    # Check for degenerate repetition (same sentence repeated)
    sentences = assistant_msg.split(".")
    if len(sentences) > 3:
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(unique_sentences) < len(sentences) * 0.5:
            return False
    return True


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)

    examples = list(read_jsonl(args.input))
    print(f"Loaded {len(examples)} teacher outputs")

    # Apply quality filter
    filtered = [
        ex for ex in examples
        if quality_filter(ex, args.min_response_words, args.max_response_words)
    ]
    removed = len(examples) - len(filtered)
    print(f"Quality filter: kept {len(filtered)}, removed {removed} "
          f"({removed / max(len(examples), 1):.0%})")

    rng.shuffle(filtered)

    total_needed = args.train + args.valid
    if len(filtered) < total_needed:
        print(f"WARNING: Only {len(filtered)} examples after filtering, "
              f"need {total_needed}. Using all available.")
        args.train = int(len(filtered) * 0.8)
        args.valid = len(filtered) - args.train

    train_rows = filtered[:args.train]
    valid_rows = filtered[args.train:args.train + args.valid]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)

    cat_dist = dict(Counter(r.get("category", "unknown") for r in train_rows))
    manifest = {
        "source": "teacher_distillation",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "quality_filters": {
            "min_response_words": args.min_response_words,
            "max_response_words": args.max_response_words,
        },
        "counts": {
            "input_total": len(examples),
            "after_filter": len(filtered),
            "removed": removed,
            "train": len(train_rows),
            "valid": len(valid_rows),
        },
        "category_distribution": cat_dist,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\nDistillation data written to {out_dir}")
    print(f"  Train: {len(train_rows)} examples")
    print(f"  Valid: {len(valid_rows)} examples")
    print(f"  Categories: {cat_dist}")


if __name__ == "__main__":
    main()
