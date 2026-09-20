"""Conciseness evaluation: SFT vs DPO-concise (and base for reference).

Positive counterpart to eval_judge.py. Shows DPO moving a SUBJECTIVE axis
(conciseness) without tanking objective correctness:

  1. Mean answer length  -- words + chars per model (expect DPO < SFT).
  2. Pairwise conciseness -- blinded, position-randomized gpt-5.5 vote:
       "which answer is more concise while remaining correct and complete?"
       reported as DPO-concise win/tie/loss vs SFT.
  3. Correctness/actionability hold-check -- the same absolute 1-5 judge from
     eval_judge.py, to confirm DPO-concise stays correct (the win must be
     "shorter, still correct", not "shorter because it dropped content").

Models loaded one at a time and freed (OOM-safe on a 24GB A30). Judge calls
go through OpenRouter in parallel.

Run from code/ AFTER DPO training:
    python -m chapter08.eval_concise
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common.env  # noqa: F401  (loads OPENROUTER_API_KEY from code/.env)
from common.jsonl import read_jsonl, write_jsonl
from common.openrouter import chat
from common.hub import split_ref, subfolder_kwargs
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE = "openai/gpt-5.5"
IT_SYSTEM = "You are an IT support assistant. Provide clear, step-by-step answers."

FMT_RE = re.compile(r"\*\*Summary:\*\*.*\*\*Steps:\*\*.*?\n\s*1\.", re.S)


def format_ok(t: str) -> bool:
    return bool(FMT_RE.search(t or ""))


# --- Absolute correctness/actionability judge (verbatim from eval_judge.py) ---
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
        r = chat([{"role": "user", "content": JUDGE_PROMPT.format(q=q[:1200], ref=ref[:1500], ans=ans[:1800])}],
                 model=JUDGE, max_tokens=2000, temperature=0.0)
        m = re.search(r"\{.*\}", r.get("content") or "", re.S)
        d = json.loads(m.group(0))
        return {k: float(d.get(k)) for k in ("correctness", "actionability", "format")}
    except Exception as e:
        return {"correctness": None, "actionability": None, "format": None, "err": repr(e)[:80]}


# --- Pairwise CONCISENESS judge: DPO-concise vs SFT, blinded A/B order ---
CONCISE_PROMPT = """You compare two IT support assistant answers to the same QUESTION on ONE axis: CONCISENESS. Pick the answer that is more concise (shorter, less padding, no redundant caveats or repetition) WHILE remaining correct and complete. If one answer is shorter but drops necessary steps or becomes incorrect/incomplete, it should NOT win. If they are equally concise and complete, answer tie.
Return ONLY compact JSON: {{"winner":"A"}} or {{"winner":"B"}} or {{"winner":"tie"}}.

QUESTION: {q}

ANSWER A: {a}

ANSWER B: {b}"""


def judge_concise(q, ans_a, ans_b):
    try:
        r = chat([{"role": "user", "content": CONCISE_PROMPT.format(
                     q=q[:1200], a=ans_a[:1800], b=ans_b[:1800])}],
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
    ap.add_argument("--dpo", default="chapter08/runs/dpo_concise",
                    help="DPO model: accepts a local path, an HF repo id, or "
                         "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch8-dpo)")
    ap.add_argument("--out", default="chapter08/eval/concise_report.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true")
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

    cache_path = Path(args.out).with_name("concise_generations_cache.json")
    if args.use_cache and cache_path.exists():
        print(f"loading cached generations from {cache_path}...", flush=True)
        gens = json.load(open(cache_path))
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

    # --- Lens 1: length ---
    def words(s):
        return len(s.split())
    length = {}
    for mdl in gens:
        w = [words(gens[mdl][i]) for i in range(n)]
        c = [len(gens[mdl][i]) for i in range(n)]
        length[mdl] = {"mean_words": round(sum(w) / n, 1), "mean_chars": round(sum(c) / n, 1)}

    # --- Lens 3: absolute correctness/actionability hold-check (all models) ---
    print("judging absolute correctness/actionability via gpt-5.5...", flush=True)
    jobs = [(mdl, i) for mdl in gens for i in range(n)]
    judge = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, qs[i], refs[i], gens[mdl][i]): (mdl, i) for mdl, i in jobs}
        for fut in as_completed(futs):
            judge[futs[fut]] = fut.result()

    # --- Lens 2: pairwise conciseness DPO-concise vs SFT, blinded ---
    print("judging pairwise CONCISENESS DPO-vs-SFT via gpt-5.5...", flush=True)
    pair_assign = {i: ("dpo_is_A" if rng.random() < 0.5 else "dpo_is_B") for i in range(n)}
    pair_raw = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i in range(n):
            if pair_assign[i] == "dpo_is_A":
                a, b = gens["dpo"][i], gens["sft"][i]
            else:
                a, b = gens["sft"][i], gens["dpo"][i]
            futs[ex.submit(judge_concise, qs[i], a, b)] = i
        for fut in as_completed(futs):
            pair_raw[futs[fut]] = fut.result()

    pair_result = {}
    dpo_wins = dpo_losses = ties = pair_errs = 0
    for i in range(n):
        r = pair_raw[i]
        if isinstance(r, dict):
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
    win_rate_excl = round(dpo_wins / (dpo_wins + dpo_losses), 4) if (dpo_wins + dpo_losses) else None

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    report = {"num_examples": n, "judge_model": JUDGE, "gen_model": BASE, "seed": args.seed,
              "models": {}}
    for mdl in gens:
        report["models"][mdl] = {
            "mean_words": length[mdl]["mean_words"],
            "mean_chars": length[mdl]["mean_chars"],
            "format_adherence": round(sum(format_ok(gens[mdl][i]) for i in range(n)) / n, 3),
            "judge_correctness": mean([judge[(mdl, i)].get("correctness") for i in range(n)]),
            "judge_actionability": mean([judge[(mdl, i)].get("actionability") for i in range(n)]),
            "judge_format": mean([judge[(mdl, i)].get("format") for i in range(n)]),
            "judge_errors": sum(1 for i in range(n) if judge[(mdl, i)].get("correctness") is None),
        }

    sft_w = length["sft"]["mean_words"]
    dpo_w = length["dpo"]["mean_words"]
    report["length_reduction_dpo_vs_sft"] = {
        "sft_mean_words": sft_w,
        "dpo_mean_words": dpo_w,
        "abs_word_reduction": round(sft_w - dpo_w, 1),
        "pct_word_reduction": round((sft_w - dpo_w) / sft_w * 100, 1) if sft_w else None,
    }
    report["pairwise_conciseness_dpo_vs_sft"] = {
        "dpo_wins": dpo_wins, "ties": ties, "dpo_losses": dpo_losses,
        "errors": pair_errs, "decided": decided,
        "dpo_win_rate_incl_ties": win_rate,
        "dpo_win_rate_excl_ties": win_rate_excl,
        "note": "Blinded A/B; DPO placed in a random slot per question; winner mapped back.",
    }
    report["holdcheck_correctness"] = {
        "sft_correctness": report["models"]["sft"]["judge_correctness"],
        "dpo_correctness": report["models"]["dpo"]["judge_correctness"],
        "sft_actionability": report["models"]["sft"]["judge_actionability"],
        "dpo_actionability": report["models"]["dpo"]["judge_actionability"],
    }

    per_example = []
    for i in range(n):
        per_example.append({
            "category": cats[i],
            "question": qs[i][:300],
            **{f"{mdl}_words": len(gens[mdl][i].split()) for mdl in gens},
            **{f"{mdl}_answer": gens[mdl][i][:900] for mdl in gens},
            **{f"{mdl}_judge": judge[(mdl, i)] for mdl in gens},
            "conc_dpo_position": "A" if pair_assign[i] == "dpo_is_A" else "B",
            "conc_raw_winner": pair_raw[i] if not isinstance(pair_raw[i], dict) else "err",
            "conc_result": pair_result[i],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    write_jsonl(out.with_name("concise_examples.jsonl"), per_example)

    print("\n==== CH8 CONCISENESS REPORT (SFT vs DPO-concise) ====")
    for mdl in ("base", "sft", "dpo"):
        r = report["models"][mdl]
        print(f"{mdl:4s} words {r['mean_words']:6.1f} chars {r['mean_chars']:7.1f} | "
              f"fmt {r['format_adherence']*100:3.0f}% | correct {r['judge_correctness']} "
              f"action {r['judge_actionability']} fmt {r['judge_format']} (errs {r['judge_errors']})")
    lr = report["length_reduction_dpo_vs_sft"]
    print(f"\nLength: SFT {lr['sft_mean_words']}w -> DPO {lr['dpo_mean_words']}w "
          f"({lr['pct_word_reduction']}% shorter)")
    p = report["pairwise_conciseness_dpo_vs_sft"]
    print(f"Pairwise conciseness: DPO wins {p['dpo_wins']} / ties {p['ties']} / "
          f"losses {p['dpo_losses']} (errs {p['errors']}) | "
          f"win-rate incl-ties {p['dpo_win_rate_incl_ties']} excl-ties {p['dpo_win_rate_excl_ties']}")
    hc = report["holdcheck_correctness"]
    print(f"Hold-check correctness: SFT {hc['sft_correctness']} -> DPO {hc['dpo_correctness']} | "
          f"action SFT {hc['sft_actionability']} -> DPO {hc['dpo_actionability']}")
    print(f"\nSaved {out} + {out.with_name('concise_examples.jsonl')}")


if __name__ == "__main__":
    main()
