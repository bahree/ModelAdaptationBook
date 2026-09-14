"""Reformat the real Stack Exchange IT answers into a consistent house format
(teacher = a strong OpenRouter model). This is the SFT-target normalization step:
content stays from the real answer; only the *shape* is standardized so the SFT
model learns a consistent IT-support response format.

House format:
    **Summary:** <one-sentence diagnosis/answer>
    **Steps:**
    1. <step>  (commands in backticks; numbered, concrete)
    2. ...

Dolly retention examples are left untouched (they are the general-capability
slice, not IT-support answers).

Run from code/. With no arguments it processes BOTH splits:
    data/it_support/train.jsonl -> data/it_support_fmt/train.jsonl   (training targets)
    data/it_support/valid.jsonl -> data/it_support_fmt/valid.jsonl   (training-time validation)
The raw data/it_support/valid.jsonl stays the held-out TEST set for the evaluation
scripts, so a model's eval loss is scored on the same answer style it trains on
while its token-F1 is still scored against the original human answers.

    python scripts/reformat_it_answers.py                 # both splits
    python scripts/reformat_it_answers.py --input X --output Y  # one file

Needs OPENROUTER_API_KEY (see code/README.md). Both output files are committed to
the repo, so you only need to run this if you rebuild the dataset from source.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common.env  # noqa: F401  triggers load_dotenv(code/.env)
from common.jsonl import read_jsonl, write_jsonl
from common.openrouter import chat

TEACHER = "anthropic/claude-sonnet-4.5"

SYSTEM = (
    "You are a senior IT support engineer and technical editor. You rewrite "
    "support answers into a consistent house format. You never add facts, "
    "commands, or steps that are not supported by the original answer, and you "
    "never remove technical detail that matters. You keep all commands, paths, "
    "and code exactly as given, in backticks."
)

INSTRUCTION = """Rewrite the ANSWER below into exactly this format:

**Summary:** <one sentence that states the fix or the direct answer>
**Steps:**
1. <first concrete step>
2. <next step>
(use as many numbered steps as needed; put commands, paths, and code in backticks)

Rules:
- Preserve the original technical content and all commands/paths verbatim.
- Do not invent steps or details not present in the original answer.
- If the answer is explanatory rather than procedural, still give a one-line
  Summary, then use the numbered list for the key points.
- Be concise. Output ONLY the reformatted answer, nothing else.

QUESTION:
{q}

ANSWER:
{a}"""


def reformat_one(question: str, answer: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION.format(q=question[:1500], a=answer[:2500])},
    ]
    r = chat(msgs, model=TEACHER, max_tokens=900, temperature=0.0)
    return (r.get("content") or "").strip()


SPLITS = [("data/it_support/train.jsonl", "data/it_support_fmt/train.jsonl"),
          ("data/it_support/valid.jsonl", "data/it_support_fmt/valid.jsonl")]


def process(inp: str, out: str, limit: int, dry: bool, workers: int) -> None:
    rows = list(read_jsonl(inp))

    def qa(row):
        msgs = row["messages"]
        q = next(m["content"] for m in msgs if m["role"] == "user")
        a = next(m["content"] for m in msgs if m["role"] == "assistant")
        return q, a

    if dry:
        shown = 0
        for row in rows:
            if row.get("source") == "dolly":
                continue
            q, a = qa(row)
            print(f"\n===== [{row.get('category')}] Q: {q[:90]}")
            print(f"--- RAW: {a[:200]}")
            print(f"--- FMT: {reformat_one(q, a)[:400]}")
            shown += 1
            if shown >= (limit or 2):
                break
        return

    # IT rows to reformat (preserve original index for ordered output).
    it_idx = [i for i, r in enumerate(rows) if r.get("source") != "dolly"]
    if limit:
        it_idx = it_idx[:limit]
    new_answers = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(reformat_one, *qa(rows[i])): i for i in it_idx}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                new_answers[i] = fut.result()
            except Exception as e:
                print(f"  WARN row {i} failed ({repr(e)[:80]}); keeping raw", flush=True)
                new_answers[i] = None
            done += 1
            if done % 25 == 0:
                print(f"  reformatted {done}/{len(it_idx)}...", flush=True)

    out_rows = []
    for i, row in enumerate(rows):
        na = new_answers.get(i)
        if na:
            out_rows.append({**row, "messages": [
                (m if m["role"] != "assistant" else {"role": "assistant", "content": na})
                for m in row["messages"]]})
        else:
            out_rows.append(row)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, out_rows)
    n_ok = sum(1 for v in new_answers.values() if v)
    print(f"Wrote {out}: {len(out_rows)} rows ({n_ok}/{len(it_idx)} IT reformatted, rest passthrough)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", "--input", dest="inp", default=None,
                    help="input split from build_it_support_dataset.py; with no --input/--output both the train and "
                         "valid splits are processed (the default the READMEs rely on)")
    ap.add_argument("--out", "--output", dest="out", default=None, help="output path (required if --input is given)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dry", action="store_true", help="print, do not write")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    args = ap.parse_args()
    if (args.inp is None) != (args.out is None):
        ap.error("--input and --output must be given together (or neither, to process both default splits)")
    pairs = [(args.inp, args.out)] if args.inp else SPLITS
    for inp, out in pairs:
        process(inp, out, args.limit, args.dry, args.workers)


if __name__ == "__main__":
    main()
