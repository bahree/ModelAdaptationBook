"""LLM-as-judge evaluation for Chapter 8 (DPO), mirroring the book's 3-lens harness.

Token-F1 is blind to preference quality: a DPO model can produce a clearly
better answer (more correct, more actionable, better formatted) while its
token overlap with the raw reference barely moves. This script reuses the
exact judge machinery from ``scripts/eval_3lens.py`` (same JUDGE model,
JUDGE_PROMPT, IT_SYSTEM, format_ok regex, token_f1, greedy gen) and adds a
pairwise DPO-vs-SFT preference vote -- the headline a preference chapter
needs.

Three FULL models are evaluated on the same 50-example IT valid set used by
the token-F1 3-way (data/it_support/valid.jsonl):
  base = Qwen/Qwen3-4B-Instruct-2507
  SFT  = chapter06/runs/sft_run1
  DPO  = chapter08/runs/dpo_run1

Each model is loaded one at a time and freed before the next to avoid OOM on
a 24GB A30. Judge calls go through OpenRouter in parallel (ThreadPoolExecutor).

Run from code/ AFTER DPO training:
    python -m chapter08.eval_judge
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common.env  # noqa: F401  (loads OPENROUTER_API_KEY from code/.env)
from common.jsonl import read_jsonl, write_jsonl
from common.openrouter import chat
from common.hub import split_ref, subfolder_kwargs
from chapter05.metrics import token_f1
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Reused verbatim from scripts/eval_3lens.py so Ch8's judge matches Ch6/Ch7 ---
BASE = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE = "openai/gpt-5.5"
IT_SYSTEM = "You are an IT support assistant. Provide clear, step-by-step answers."

FMT_RE = re.compile(r"\*\*Summary:\*\*.*\*\*Steps:\*\*.*?\n\s*1\.", re.S)
def format_ok(t: str) -> bool:
    return bool(FMT_RE.search(t or ""))

JUDGE_PROMPT = """You evaluate IT support assistant answers. Given the user QUESTION and the ASSISTANT answer, rate the ASSISTANT answer 1-5 (5=best) on:
- correctness: is it technically correct and does it address the question? Judge on its OWN technical merits. A valid alternative approach is fully correct even if it differs from the reference; do NOT penalize a different-but-valid solution.
- actionability: concrete, usable steps/commands the user can follow
- format: clear, well-structured, easy to scan under time pressure
The REFERENCE is provided only as background on the topic, not as the single correct answer.
Return ONLY compact JSON: {{"correctness":N,"actionability":N,"format":N}}

QUESTION: {q}
REFERENCE (background only): {ref}
ASSISTANT: {ans}"""

def judge_one(q, ref, ans):
    try:
        # gpt-5.5 is a reasoning model: it spends part of the token budget on hidden
        # reasoning before emitting content, so a small cap returns empty content.
        # The prompt text is unchanged; only the budget is raised.
        r = chat([{"role": "user", "content": JUDGE_PROMPT.format(q=q[:1200], ref=ref[:1500], ans=ans[:1800])}],
                 model=JUDGE, max_tokens=2000, temperature=0.0)
        m = re.search(r"\{.*\}", r.get("content") or "", re.S)
        d = json.loads(m.group(0))
        return {k: float(d.get(k)) for k in ("correctness", "actionability", "format")}
    except Exception as e:
        return {"correctness": None, "actionability": None, "format": None, "err": repr(e)[:80]}


# --- Pairwise preference judge: DPO vs SFT, blinded A/B order ---
PAIRWISE_PROMPT = """You compare two IT support assistant answers to the same QUESTION and decide which is better OVERALL, weighing correctness, actionability, and clear scannable format. Judge each on its own technical merits; a valid alternative approach is fully correct even if it differs from the reference. The REFERENCE is background only, not the single correct answer.
Return ONLY compact JSON: {{"winner":"A"}} or {{"winner":"B"}} or {{"winner":"tie"}}.

QUESTION: {q}
REFERENCE (background only): {ref}

ANSWER A: {a}

ANSWER B: {b}"""

def judge_pair(q, ref, ans_a, ans_b):
    try:
        r = chat([{"role": "user", "content": PAIRWISE_PROMPT.format(
                     q=q[:1200], ref=ref[:1500], a=ans_a[:1800], b=ans_b[:1800])}],
                 model=JUDGE, max_tokens=2000, temperature=0.0)
        m = re.search(r"\{.*\}", r.get("content") or "", re.S)
        w = str(json.loads(m.group(0)).get("winner", "")).strip().lower()
        if w.startswith("a"):
            return "A"
        if w.startswith("b"):
            return "B"
        return "tie"
    except Exception as e:
        return {"err": repr(e)[:80]}


def load(model_path):
    """Load a FULL model directly (no LoRA adapter logic; all three are full models)."""
    p, s = split_ref(model_path)
    m = AutoModelForCausalLM.from_pretrained(p, **subfolder_kwargs(s), dtype=torch.bfloat16, device_map="cuda:0")
    m.eval()
    return m


def gen_all(model, tok, prompts, max_new=400):
    outs = []
    for msgs in prompts:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=False).to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default="data/it_support/valid.jsonl")
    ap.add_argument("--base", default=BASE,
                    help="Base model: accepts a local path, an HF repo id, or "
                         "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch6-sft)")
    ap.add_argument("--sft", default="chapter06/runs/sft_run1",
                    help="SFT model: accepts a local path, an HF repo id, or "
                         "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch6-sft)")
    ap.add_argument("--dpo", default="chapter08/runs/dpo_run1",
                    help="DPO model: accepts a local path, an HF repo id, or "
                         "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch8-dpo)")
    ap.add_argument("--out", default="chapter08/eval/judge_report.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true",
                    help="reuse cached model generations (skip the ~36-min regeneration)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    val = list(read_jsonl(args.valid))
    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = [[{"role": "system", "content": IT_SYSTEM},
                {"role": "user", "content": next(m["content"] for m in ex["messages"] if m["role"] == "user")}]
               for ex in val]
    refs = [next(m["content"] for m in ex["messages"] if m["role"] == "assistant") for ex in val]
    cats = [ex.get("category", "general") for ex in val]
    qs = [p[1]["content"] for p in prompts]
    n = len(val)

    # Generate one model at a time, free before next (OOM-safe on 24GB A30).
    # Generation is the expensive step (~12 min/model). Cache full outputs so the
    # judging pass can be re-run cheaply (e.g. to tune the judge token budget).
    cache_path = Path(args.out).with_name("generations_cache.json")
    if args.use_cache and cache_path.exists():
        print(f"loading cached generations from {cache_path}...", flush=True)
        gens = json.load(open(cache_path))
        assert all(len(gens[k]) == n for k in ("base", "sft", "dpo")), "cache length mismatch"
    else:
        gens = {}
        for name, path in (("base", args.base), ("sft", args.sft), ("dpo", args.dpo)):
            print(f"generating {name} ({path})...", flush=True)
            model = load(path)
            gens[name] = gen_all(model, tok, prompts)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(gens, open(cache_path, "w"))

    # --- Absolute judge scores: all (model, example) pairs in parallel ---
    print("judging absolute scores via gpt-5.5...", flush=True)
    jobs = [(mdl, i) for mdl in gens for i in range(n)]
    judge = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, qs[i], refs[i], gens[mdl][i]): (mdl, i) for mdl, i in jobs}
        for fut in as_completed(futs):
            judge[futs[fut]] = fut.result()

    # --- Pairwise DPO-vs-SFT, blinded A/B order ---
    print("judging pairwise DPO-vs-SFT via gpt-5.5...", flush=True)
    # For each example pick a random side for DPO; record the assignment.
    pair_assign = {}  # i -> "dpo_is_A" or "dpo_is_B"
    for i in range(n):
        pair_assign[i] = "dpo_is_A" if rng.random() < 0.5 else "dpo_is_B"
    pair_raw = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i in range(n):
            if pair_assign[i] == "dpo_is_A":
                a, b = gens["dpo"][i], gens["sft"][i]
            else:
                a, b = gens["sft"][i], gens["dpo"][i]
            futs[ex.submit(judge_pair, qs[i], refs[i], a, b)] = i
        for fut in as_completed(futs):
            pair_raw[futs[fut]] = fut.result()

    # Resolve A/B winner back to dpo/sft/tie.
    pair_result = {}  # i -> "dpo" | "sft" | "tie" | "err"
    dpo_wins = dpo_losses = ties = pair_errs = 0
    for i in range(n):
        r = pair_raw[i]
        if isinstance(r, dict):  # error
            pair_result[i] = "err"
            pair_errs += 1
            continue
        if r == "tie":
            pair_result[i] = "tie"
            ties += 1
        else:
            dpo_pos = "A" if pair_assign[i] == "dpo_is_A" else "B"
            if r == dpo_pos:
                pair_result[i] = "dpo"
                dpo_wins += 1
            else:
                pair_result[i] = "sft"
                dpo_losses += 1

    decided = dpo_wins + dpo_losses + ties
    win_rate = round(dpo_wins / decided, 4) if decided else None
    win_rate_excl_ties = round(dpo_wins / (dpo_wins + dpo_losses), 4) if (dpo_wins + dpo_losses) else None

    # --- Aggregate absolute scores ---
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    report = {"num_examples": n, "judge_model": JUDGE, "seed": args.seed, "models": {}}
    for mdl in gens:
        f1s = [token_f1(gens[mdl][i], refs[i]) for i in range(n)]
        fmt = [format_ok(gens[mdl][i]) for i in range(n)]
        jc = [judge[(mdl, i)].get("correctness") for i in range(n)]
        ja = [judge[(mdl, i)].get("actionability") for i in range(n)]
        jf = [judge[(mdl, i)].get("format") for i in range(n)]
        bycat = defaultdict(list)
        for i in range(n):
            bycat[cats[i]].append(f1s[i])
        report["models"][mdl] = {
            "token_f1": round(sum(f1s) / len(f1s), 4),
            "format_adherence": round(sum(fmt) / len(fmt), 3),
            "judge_correctness": mean(jc),
            "judge_actionability": mean(ja),
            "judge_format": mean(jf),
            "judge_errors": sum(1 for i in range(n) if judge[(mdl, i)].get("correctness") is None),
            "token_f1_by_category": {c: round(sum(v) / len(v), 4) for c, v in bycat.items()},
        }

    report["pairwise_dpo_vs_sft"] = {
        "dpo_wins": dpo_wins,
        "ties": ties,
        "dpo_losses": dpo_losses,
        "errors": pair_errs,
        "decided": decided,
        "dpo_win_rate_incl_ties": win_rate,
        "dpo_win_rate_excl_ties": win_rate_excl_ties,
        "note": "DPO placed in a random A/B slot per question to remove position bias; "
                "winner mapped back via recorded assignment.",
    }

    # --- Per-example dump so the chapter can show without re-running ---
    per_example = []
    for i in range(n):
        per_example.append({
            "category": cats[i],
            "question": qs[i][:300],
            "reference": refs[i][:400],
            **{f"{mdl}_answer": gens[mdl][i][:800] for mdl in gens},
            **{f"{mdl}_judge": judge[(mdl, i)] for mdl in gens},
            "pairwise_dpo_position": "A" if pair_assign[i] == "dpo_is_A" else "B",
            "pairwise_raw_winner": pair_raw[i] if not isinstance(pair_raw[i], dict) else "err",
            "pairwise_result": pair_result[i],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    write_jsonl(out.with_name("judge_examples.jsonl"), per_example)

    # --- Console summary ---
    print("\n==== CH8 LLM-AS-JUDGE REPORT ====")
    for mdl in ("base", "sft", "dpo"):
        r = report["models"][mdl]
        print(f"{mdl:5s}  tokenF1 {r['token_f1']:.3f} | format {r['format_adherence']*100:4.0f}% | "
              f"judge correct {r['judge_correctness']} action {r['judge_actionability']} "
              f"format {r['judge_format']} (errs {r['judge_errors']})")
    p = report["pairwise_dpo_vs_sft"]
    print(f"\nPairwise DPO-vs-SFT: DPO wins {p['dpo_wins']} / ties {p['ties']} / "
          f"losses {p['dpo_losses']} (errors {p['errors']})")
    print(f"  DPO win-rate incl ties: {p['dpo_win_rate_incl_ties']}  "
          f"excl ties: {p['dpo_win_rate_excl_ties']}")
    print(f"\nSaved {out} + {out.with_name('judge_examples.jsonl')}")


if __name__ == "__main__":
    main()
