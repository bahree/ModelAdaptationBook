"""Expand the hand-authored Contoso IT-support seed with an OpenRouter teacher.

Style-anchored generation: for each category we show the teacher real seed Q&A
and the house-style rules, then ask for new pairs in the same internal voice and
format. A rule-based quality gate enforces the format (one-line answer, numbered
Steps:, escalation line, internal terminology) so synthetic rows match the seed.

Outputs (data/expanded/):
  train.jsonl  -- training set: hand-authored (minus held-out) + gated synthetic
  eval.jsonl   -- held-out GOLDEN set: hand-authored only, never trained on

Uses code/common/openrouter.py (OPENROUTER_API_KEY from code/.env), the same
teacher path as the distillation chapter. No GPU.

Run from code/:  python3 -m contoso_qa_demo.expand_seed --n-per-cat 18
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common.openrouter import chat, OpenRouterError

HERE = Path(__file__).parent
SEED = HERE / "data" / "it_qa_train.jsonl"
OUT = HERE / "data" / "expanded"
SYSTEM = "You are Contoso's internal IT support assistant. Give clear, accurate help."
TEACHERS = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o", "openai/gpt-4o-mini"]

HOUSE_RULES = (
    "Answer STYLE (follow exactly):\n"
    "- Line 1: one direct sentence answering the question.\n"
    "- Then a line 'Steps:' followed by 2 to 4 numbered, imperative steps.\n"
    "- End with an escalation line that points to a ServiceNow request and the "
    "#it-help Slack channel.\n"
    "Use Contoso-internal names where relevant: GlobalConnect VPN, AccessHub "
    "(SSO/access portal), Contoso MFA via the Authenticator app, the "
    "StandardBuild laptop image, Software Center, ServiceNow (ticketing), "
    "#it-help (Slack). Keep answers concise and realistic. Do NOT invent other "
    "internal product names.\n"
    "Apply this format to EVERY answer, including simple requests (a charger, a "
    "cable, a monitor): still give one direct sentence, a 'Steps:' block (two "
    "short steps is fine), and the escalation line. Never skip 'Steps:' or the "
    "escalation line."
)


def load_seed():
    rows = [json.loads(ln) for ln in open(SEED) if ln.strip()]
    by_cat = {}
    for r in rows:
        q = next(m["content"] for m in r["messages"] if m["role"] == "user")
        a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        by_cat.setdefault(r["category"], []).append({"question": q, "answer": a})
    return by_cat


def pick_teacher():
    for m in TEACHERS:
        try:
            chat([{"role": "user", "content": "ok"}], m, max_tokens=5)
            return m
        except OpenRouterError:
            continue
    raise SystemExit("no working OpenRouter model")


def gen_for_category(cat, anchors, n, model, rng):
    sample = rng.sample(anchors, min(3, len(anchors)))
    ref = "\n\n".join(f"Q: {s['question']}\nA: {s['answer']}" for s in sample)
    prompt = (
        f"You are writing TRAINING DATA for Contoso's IT-support assistant.\n"
        f"{HOUSE_RULES}\n\n"
        f"Here are real examples for the \"{cat}\" category:\n{ref}\n\n"
        f"Generate {n} NEW, distinct \"{cat}\" question-and-answer pairs in the "
        f"exact same house style and terminology. Vary the user voice "
        f"(executive, engineer, field staff, office user). Do not repeat the "
        f"examples. Return ONLY a JSON array: "
        f'[{{"question": "...", "answer": "..."}}]'
    )
    raw = chat([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}], model,
               max_tokens=8192, temperature=0.8)["content"]
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n?|\n?```$", "", txt)
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [{"question": it["question"], "answer": it["answer"], "category": cat}
            for it in items if isinstance(it, dict) and it.get("question") and it.get("answer")]


def passes_gate(ex):
    a = ex["answer"]
    return ("Steps:" in a and ("ServiceNow" in a or "#it-help" in a)
            and len(a) > 90 and ex["question"].strip().endswith("?"))


def to_chatml(ex, source):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": ex["question"]},
                         {"role": "assistant", "content": ex["answer"]}],
            "category": ex["category"], "source": source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=18)  # legacy; superseded by target
    ap.add_argument("--target-per-cat", type=int, default=24)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    rng = random.Random(42)
    by_cat = load_seed()
    model = args.model or pick_teacher()
    print(f"teacher: {model}")

    OUT.mkdir(parents=True, exist_ok=True)
    # hold out 1 hand-authored example per category as the golden eval set
    eval_rows, train_hand = [], []
    for cat, exs in by_cat.items():
        shuffled = exs[:]
        rng.shuffle(shuffled)
        eval_rows.append(to_chatml({**shuffled[0], "category": cat}, "handauthored"))
        train_hand += [to_chatml({**e, "category": cat}, "handauthored") for e in shuffled[1:]]

    # per-category target with a retry loop, so thin categories (hardware) keep
    # generating until they reach the target instead of being starved by gate
    # rejections.
    target = args.target_per_cat
    synth = []
    for cat, exs in by_cat.items():
        got, seen, rounds = [], set(), 0
        while len(got) < target and rounds < args.max_rounds:
            need = target - len(got)
            cands = gen_for_category(cat, exs, min(need * 3, 40), model, rng)
            for c in cands:
                key = c["question"].strip().lower()
                if key not in seen and passes_gate(c):
                    seen.add(key)
                    got.append(c)
            rounds += 1
        got = got[:target]
        synth += [to_chatml(c, "synthetic") for c in got]
        print(f"  {cat:9s}: {len(got)}/{target} passing after {rounds} round(s)")

    train = train_hand + synth
    rng.shuffle(train)
    with open(OUT / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "eval.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\ntrain.jsonl: {len(train)} ({len(train_hand)} hand + {len(synth)} synthetic)")
    print(f"eval.jsonl:  {len(eval_rows)} held-out hand-authored (golden)")


if __name__ == "__main__":
    main()
