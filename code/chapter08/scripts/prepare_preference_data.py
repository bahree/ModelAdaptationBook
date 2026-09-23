"""Create preference pairs for DPO training from real human-graded data.

PRIMARY PATH (default): reformat the dataset's real, human-graded
preference pairs into TRL conversational DPO format.

The source file ``data/it_support/preferences.jsonl`` ships 300 pairs in a
flat string schema drawn from Stack Exchange (Super User, Ask Ubuntu,
Server Fault).  Each pair contrasts a high-score (community-accepted)
answer against a low-score answer to the *same* question::

    {"prompt": "<user question>",
     "chosen": "<high-score answer>",
     "rejected": "<low-score answer>"}

Because the "chosen" answer is a human-graded *better* answer (not a
synthetic SFT-model generation), the preference signal is reliable and
DPO has something real to learn.  This replaces the earlier synthetic
approach (SFT-chosen vs base-rejected), where the weak SFT teacher's
answers were not reliably better than the base model's.

This script reformats each flat pair into the conversational schema that
``train_dpo.py`` (TRL DPOTrainer) expects::

    {"prompt":   [{"role": "system",    "content": <SYSTEM_PROMPT>},
                  {"role": "user",       "content": <question>}],
     "chosen":   [{"role": "assistant", "content": <high-score answer>}],
     "rejected": [{"role": "assistant", "content": <low-score answer>}]}

Contamination guard: any preference pair whose user prompt also appears in
the validation split (``data/it_support/valid.jsonl``) or the held-out test split
(``data/it_support/test.jsonl``, the one the three-way eval reports on) is dropped
so the DPO training data never overlaps the evaluation prompts.

Run from code/:
    python -m chapter08.scripts.prepare_preference_data \
        --preferences data/it_support/preferences.jsonl \
        --eval_valid data/it_support/valid.jsonl \
        --out chapter08/data/preference_pairs \
        --num_valid 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path

from common.jsonl import read_jsonl, write_jsonl
from common.manifest import write_json
from common.seed import seed_everything

SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def _eval_prompts(eval_valid: str) -> set[str]:
    """Collect the user prompts of every evaluation split: the file given plus a sibling
    test.jsonl when one exists (the held-out split the three-way eval reports on)."""
    prompts: set[str] = set()
    files = [Path(eval_valid)]
    test_file = Path(eval_valid).parent / "test.jsonl"
    if test_file.exists() and test_file != files[0]:
        files.append(test_file)
    for f in files:
        for ex in read_jsonl(f):
            msgs = ex["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            prompts.add(user_msg.strip())
    return prompts


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build DPO preference pairs from real human-graded data"
    )
    ap.add_argument(
        "--preferences", default="data/it_support/preferences.jsonl",
        help="Flat human-graded preference pairs (prompt/chosen/rejected)",
    )
    ap.add_argument(
        "--eval_valid", default="data/it_support/valid.jsonl",
        help="Validation split; its prompts and those of the sibling test.jsonl are excluded (contamination guard)",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--num_valid", type=int, default=30,
        help="Number of pairs held out for DPO validation (rest are train)",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    raw = list(read_jsonl(args.preferences))
    print(f"Loaded {len(raw)} raw preference pairs from {args.preferences}")

    # Contamination guard: drop any pair whose prompt is in the eval set.
    eval_prompts = _eval_prompts(args.eval_valid)
    print(f"Eval set has {len(eval_prompts)} unique prompts (excluded)")

    pairs = []
    dropped_contaminated = 0
    dropped_empty = 0
    for rec in raw:
        prompt = (rec.get("prompt") or "").strip()
        chosen = (rec.get("chosen") or "").strip()
        rejected = (rec.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            dropped_empty += 1
            continue
        if prompt in eval_prompts:
            dropped_contaminated += 1
            continue
        pairs.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "chosen": [
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "assistant", "content": rejected},
            ],
        })

    print(f"Dropped {dropped_contaminated} contaminated pairs (prompt in eval set)")
    print(f"Dropped {dropped_empty} empty/malformed pairs")
    print(f"Kept {len(pairs)} usable preference pairs")

    # Deterministic shuffle, then split train/valid.
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    num_valid = min(args.num_valid, len(pairs))
    valid_pairs = pairs[:num_valid]
    train_pairs = pairs[num_valid:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_pairs)
    write_jsonl(out_dir / "valid.jsonl", valid_pairs)

    manifest = {
        "source": "real_human_graded_preference_pairs",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_file": args.preferences,
        "system_prompt": SYSTEM_PROMPT,
        # chosen/rejected come from the same human-graded source file;
        # kept as keys for downstream tooling/test compatibility.
        "chosen_model": "human_graded_high_score_answer",
        "rejected_model": "human_graded_low_score_answer",
        "raw_pairs": len(raw),
        "dropped_contaminated": dropped_contaminated,
        "dropped_empty": dropped_empty,
        "counts": {"train": len(train_pairs), "valid": len(valid_pairs)},
        "seed": args.seed,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\nPreference data written to {out_dir}")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Valid: {len(valid_pairs)} pairs")


if __name__ == "__main__":
    main()
