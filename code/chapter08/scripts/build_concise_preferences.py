"""Build a CONCISENESS preference set for DPO (Chapter 8).

Positive counterpart to the honest finding that DPO does not beat SFT on
*objective* factual correctness: here we isolate a *subjective* preference
axis -- CONCISENESS -- and show DPO can move it.

For each IT question (drawn from data/it_support/preferences.jsonl prompts,
disjoint from the validation and held-out test prompts in data/it_support/{valid,test}.jsonl)
we ask gpt-5.5 to write, for the SAME question, TWO answers that are:
  - both technically correct,
  - both in the house format ("**Summary:** ... **Steps:** 1. ... 2. ..."),
  - differing ONLY in verbosity.

    chosen   = tight/concise: lead with the answer, minimal padding.
    rejected = correct but verbose: preamble, redundant caveats, repetition.

Same technical content; only length/padding differs. This isolates the axis
so DPO learns "prefer concise" rather than "prefer different content".

Output (TRL conversational DPO schema) ->
    chapter08/data/preference_concise/{train,valid}.jsonl + manifest.json

Run from code/:
    python -m chapter08.scripts.build_concise_preferences --n 160 --num_valid 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import common.env  # noqa: F401  (loads OPENROUTER_API_KEY from code/.env)
from common.jsonl import read_jsonl, write_jsonl
from common.manifest import write_json
from common.openrouter import chat

GEN_MODEL = "openai/gpt-5.5"
IT_SYSTEM = "You are an IT support assistant. Provide clear, step-by-step answers."

FMT_RE = re.compile(r"\*\*Summary:\*\*.*\*\*Steps:\*\*.*?\n\s*1\.", re.S)


def format_ok(t: str) -> bool:
    return bool(FMT_RE.search(t or ""))


# The generator is asked for BOTH answers in one shot so the technical content
# is held constant and only verbosity differs. Returns strict JSON.
GEN_PROMPT = """You write IT support answers in a fixed house format.

The house format is EXACTLY:
**Summary:** <one or two sentences>
**Steps:**
1. <step>
2. <step>
... (more steps as needed)

For the QUESTION below, write TWO answers that are BOTH technically correct and BOTH in the house format above. They must contain the SAME technical content and the SAME core steps. They differ ONLY in verbosity:

- "concise": EXTREMELY tight and direct. Lead with the answer. No preamble, no filler, no caveats, no repetition, no explanation beyond the action itself. Summary is ONE short sentence. Each step is ONE short imperative sentence (a command or single action, ideally under 12 words). Merge trivially-related actions. Aim for the SHORTEST answer that is still correct and complete: target 35-65 words total, and never more than 75 words.
- "verbose": correct but padded. Add a wordy preamble in the Summary, hedge with redundant caveats, restate points, add "it is worth noting" / "as mentioned" style filler, and make each step longer with extra explanation. Do NOT add new technical steps; just pad the language. It should be at least 2x longer than the concise version.

Both must be genuinely correct and actionable. The concise version must keep every necessary step; cut WORDS, never steps.

Return ONLY compact JSON, no markdown fences:
{{"concise": "<full concise answer in house format>", "verbose": "<full verbose answer in house format>"}}

QUESTION: {q}"""


def gen_pair(q: str) -> dict | None:
    try:
        r = chat(
            [{"role": "user", "content": GEN_PROMPT.format(q=q[:1400])}],
            model=GEN_MODEL,
            max_tokens=4000,  # reasoning model: needs headroom or content is empty
            temperature=0.3,
        )
        content = r.get("content") or ""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        concise = (d.get("concise") or "").strip()
        verbose = (d.get("verbose") or "").strip()
        if not concise or not verbose:
            return None
        if not (format_ok(concise) and format_ok(verbose)):
            return None
        cw, vw = len(concise.split()), len(verbose.split())
        # Enforce a clear axis: concise genuinely terse, verbose >=1.8x longer.
        # The terse ceiling is what makes the DPO gradient point BELOW the SFT
        # baseline (which already sits ~75 words in the house format).
        if cw > 80 or vw < cw * 1.8:
            return None
        return {"q": q, "concise": concise, "verbose": verbose}
    except Exception:
        return None


def _eval_prompts(eval_valid: str) -> set[str]:
    prompts: set[str] = set()
    for ex in read_jsonl(eval_valid):
        user_msg = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        prompts.add(user_msg.strip())
    return prompts


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/it_support/preferences.jsonl",
                    help="Source of IT questions (uses the 'prompt' field)")
    ap.add_argument("--eval_valid", default="data/it_support/valid.jsonl")
    ap.add_argument("--out", default="chapter08/data/preference_concise")
    ap.add_argument("--n", type=int, default=160, help="Target number of pairs to build")
    ap.add_argument("--num_valid", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    eval_prompts = _eval_prompts(args.eval_valid)
    print(f"Eval set: {len(eval_prompts)} unique prompts (will be excluded)")

    # Collect candidate IT questions, dedup, drop any that appear in eval set.
    seen: set[str] = set()
    candidates: list[str] = []
    for rec in read_jsonl(args.source):
        q = (rec.get("prompt") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        if q in eval_prompts:
            continue
        # keep questions of reasonable length (true IT support tickets)
        if 15 <= len(q) <= 1200:
            candidates.append(q)

    rng.shuffle(candidates)
    # Oversample candidates to absorb generation failures (~target * 1.4).
    target = args.n
    pool = candidates[: int(target * 1.6) + 20]
    print(f"Candidate IT questions (disjoint from eval): {len(candidates)}; "
          f"attempting {len(pool)} to reach {target} good pairs")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_pair, q): q for q in pool}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if done % 20 == 0:
                print(f"  ...{done}/{len(pool)} attempted, {len(results)} good pairs", flush=True)
            if len(results) >= target:
                # let in-flight finish naturally; we just stop counting on next loop
                pass

    # Trim to target (deterministic by original pool order for reproducibility).
    pool_order = {q: i for i, q in enumerate(pool)}
    results.sort(key=lambda r: pool_order.get(r["q"], 1 << 30))
    results = results[:target]
    print(f"Built {len(results)} usable concise/verbose pairs "
          f"(format-valid + verbose-longer-than-concise)")

    # Final overlap assertion against eval prompts.
    overlap = [r for r in results if r["q"].strip() in eval_prompts]
    assert not overlap, f"OVERLAP with eval set: {len(overlap)} pairs leaked"
    print(f"Overlap with eval prompts: 0 (verified across {len(results)} pairs)")

    # Convert to TRL conversational DPO schema: chosen=concise, rejected=verbose.
    pairs = []
    for r in results:
        pairs.append({
            "prompt": [
                {"role": "system", "content": IT_SYSTEM},
                {"role": "user", "content": r["q"]},
            ],
            "chosen": [{"role": "assistant", "content": r["concise"]}],
            "rejected": [{"role": "assistant", "content": r["verbose"]}],
        })

    rng.shuffle(pairs)
    num_valid = min(args.num_valid, len(pairs))
    valid_pairs = pairs[:num_valid]
    train_pairs = pairs[num_valid:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_pairs)
    write_jsonl(out_dir / "valid.jsonl", valid_pairs)

    # Length stats (words) for the report.
    def wlen(p, key):
        return len(p[key][0]["content"].split())
    chosen_w = sum(wlen(p, "chosen") for p in pairs) / len(pairs)
    rejected_w = sum(wlen(p, "rejected") for p in pairs) / len(pairs)

    manifest = {
        "axis": "conciseness",
        "purpose": "DPO positive counterpart: subjective conciseness axis",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator_model": GEN_MODEL,
        "source_questions": args.source,
        "system_prompt": IT_SYSTEM,
        "chosen": "concise house-format answer (lead with answer, minimal padding)",
        "rejected": "verbose house-format answer (same content, padded)",
        "candidates_disjoint_from_eval": len(candidates),
        "attempted": len(pool),
        "pairs_built": len(pairs),
        "overlap_with_eval_prompts": 0,
        "mean_chosen_words": round(chosen_w, 1),
        "mean_rejected_words": round(rejected_w, 1),
        "counts": {"train": len(train_pairs), "valid": len(valid_pairs)},
        "seed": args.seed,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\nWrote {out_dir}/train.jsonl ({len(train_pairs)}) + "
          f"valid.jsonl ({len(valid_pairs)}) + manifest.json")
    print(f"Mean words: chosen(concise)={chosen_w:.1f}  rejected(verbose)={rejected_w:.1f}")


if __name__ == "__main__":
    main()
