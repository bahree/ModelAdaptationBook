"""Top up a single thin category (default: hardware) and append to the expanded
training set, without regenerating the other categories.

Hardware answers (short requests like a charger or a cable) resist the house
format, so the gate rejects most. We overgenerate with a retry loop, dedupe
against what already exists, and append until we hit the target.

Run from code/:  python3 -m contoso_qa_demo.topup_hardware --cat hardware --add 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contoso_qa_demo.expand_seed import (load_seed, gen_for_category, passes_gate,
                                       to_chatml, pick_teacher)

HERE = Path(__file__).parent
TRAIN = HERE / "data" / "expanded" / "train.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="hardware")
    ap.add_argument("--add", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    import random
    rng = random.Random(7)

    rows = [json.loads(ln) for ln in open(TRAIN) if ln.strip()]
    existing_q = {next(m["content"] for m in r["messages"] if m["role"] == "user").strip().lower()
                  for r in rows}
    before = sum(1 for r in rows if r["category"] == args.cat)
    anchors = load_seed()[args.cat]
    model = args.model or pick_teacher()
    print(f"teacher: {model} | {args.cat} before: {before} | target +{args.add}")

    new, seen, rounds = [], set(existing_q), 0
    while len(new) < args.add and rounds < args.max_rounds:
        need = args.add - len(new)
        for c in gen_for_category(args.cat, anchors, min(need * 4, 40), model, rng):
            k = c["question"].strip().lower()
            if k not in seen and passes_gate(c):
                seen.add(k)
                new.append(c)
        rounds += 1
        print(f"  round {rounds}: {len(new)}/{args.add} new {args.cat} so far")

    rows += [to_chatml(c, "synthetic") for c in new[:args.add]]
    rng.shuffle(rows)
    with open(TRAIN, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    after = sum(1 for r in rows if r["category"] == args.cat)
    print(f"{args.cat}: {before} -> {after} | train total: {len(rows)}")


if __name__ == "__main__":
    main()
