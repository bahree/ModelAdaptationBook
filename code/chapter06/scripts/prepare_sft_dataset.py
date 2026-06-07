"""Prepare a technical-support subset of Dolly 15K for SFT training.

Filters by category and keyword, converts to chat-message JSONL,
and writes train/val splits with a manifest for reproducibility.

Run from code/:
    python -m chapter06.scripts.prepare_sft_dataset \
        --out chapter06/data/dolly_sft
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from common.jsonl import write_jsonl
from common.manifest import write_json
from common.seed import seed_everything

# Dolly categories that map well to technical support
SUPPORT_CATEGORIES = [
    "closed_qa",
    "open_qa",
    "information_extraction",
    "summarization",
]

# Surface-level filter for technical content
TECHNICAL_KEYWORDS = [
    "how do", "how to", "what is", "configure", "install",
    "setup", "error", "troubleshoot", "fix", "solve",
    "computer", "software", "system", "network", "server",
    "database", "application", "program", "code", "deploy",
]

SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def is_technical(text: str) -> bool:
    """Return True if *text* contains at least one technical keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in TECHNICAL_KEYWORDS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare Dolly 15K technical subset for SFT."
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--train", type=int, default=450, help="Training examples")
    ap.add_argument("--valid", type=int, default=50, help="Validation examples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min_length", type=int, default=40,
        help="Minimum character length for instruction+response",
    )
    ap.add_argument(
        "--max_length", type=int, default=2000,
        help="Maximum character length for instruction+response",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Dolly 15K dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    # --- Filter for technical support examples ---
    candidates = []
    for row in ds:
        if row.get("category") not in SUPPORT_CATEGORIES:
            continue
        instruction = row.get("instruction", "")
        context = row.get("context", "")
        response = row.get("response", "")
        total_len = len(instruction) + len(context) + len(response)
        if not (args.min_length <= total_len <= args.max_length):
            continue
        if not is_technical(instruction) and not is_technical(response):
            continue
        candidates.append(row)

    print(f"Filtered to {len(candidates)} technical examples")
    rng.shuffle(candidates)

    total_needed = args.train + args.valid
    if len(candidates) < total_needed:
        raise RuntimeError(
            f"Need {total_needed} examples but only found {len(candidates)}"
        )

    # --- Convert to chat-message format ---
    def to_messages(row):
        user_content = row["instruction"]
        ctx = row.get("context", "").strip()
        if ctx:
            user_content = f"{ctx}\n\n{user_content}"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row["response"]},
            ],
            "category": row.get("category", "unknown"),
        }

    all_rows = [to_messages(c) for c in candidates[:total_needed]]
    train_rows = all_rows[: args.train]
    valid_rows = all_rows[args.train : args.train + args.valid]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)

    # --- Manifest for reproducibility ---
    cat_dist = dict(Counter(r["category"] for r in train_rows))
    manifest = {
        "dataset": "databricks/databricks-dolly-15k",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "filters": {
            "categories": SUPPORT_CATEGORIES,
            "min_length": args.min_length,
            "max_length": args.max_length,
        },
        "counts": {"train": len(train_rows), "valid": len(valid_rows)},
        "category_distribution": cat_dist,
        "system_prompt": SYSTEM_PROMPT,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\nDataset written to {out_dir}")
    print(f"  Train : {len(train_rows)} examples")
    print(f"  Valid : {len(valid_rows)} examples")
    print(f"  Categories: {cat_dist}")


if __name__ == "__main__":
    main()
