"""Evaluate the LoRA-DPO adapter against SFT and full-parameter DPO.

Mirrors chapter08/eval_judge.py exactly (same BASE/JUDGE/IT_SYSTEM, same
JUDGE_PROMPT/PAIRWISE_PROMPT/format_ok, same greedy gen, same 50-example IT
valid set) but adds a fourth model: the LoRA-DPO adapter loaded on top of the
SFT checkpoint via PeftModel. It answers the chapter's single-card question:

  - token-F1 for base / SFT / full-DPO / LoRA-DPO side by side
  - LLM-judge absolute (correctness/actionability/format) for LoRA-DPO
  - pairwise blinded LoRA-DPO vs SFT (did the adapter improve like full-DPO did?)
  - pairwise blinded LoRA-DPO vs full-DPO (do they match?)

Each model is loaded one at a time and freed before the next (OOM-safe on a
single 24 GB A30). Judge calls go through OpenRouter in parallel.

Run from code/ AFTER LoRA-DPO training:
    CUDA_VISIBLE_DEVICES=0 python -m chapter08.eval_lora_dpo
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common.env  # noqa: F401  (loads OPENROUTER_API_KEY from code/.env)
from common.jsonl import read_jsonl, write_jsonl
from common.hub import split_ref, subfolder_kwargs
from chapter05.metrics import token_f1
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Reuse the exact judge machinery from eval_judge.py so this matches the
# full-parameter run lens for lens.
from chapter08.eval_judge import (
    BASE, JUDGE, IT_SYSTEM, format_ok, judge_one, judge_pair, gen_all,
)


def load_full(model_path):
    p, s = split_ref(model_path)
    m = AutoModelForCausalLM.from_pretrained(p, **subfolder_kwargs(s), dtype=torch.bfloat16, device_map="cuda:0")
    m.eval()
    return m


def load_lora(sft_path, adapter_path):
    """SFT base + LoRA-DPO adapter, exactly the inference path a reader would use."""
    sft_p, sft_s = split_ref(sft_path)
    base = AutoModelForCausalLM.from_pretrained(sft_p, **subfolder_kwargs(sft_s), dtype=torch.bfloat16, device_map="cuda:0")
    adapter_p, adapter_s = split_ref(adapter_path)
    m = PeftModel.from_pretrained(base, adapter_p, **subfolder_kwargs(adapter_s))
    m.eval()
    return m


def pairwise(gens_x, gens_y, qs, refs, rng, n, workers, label_x, label_y):
    """Blinded, position-randomized pairwise vote: x vs y. Returns dict + per-i result."""
    assign = {i: ("x_is_A" if rng.random() < 0.5 else "x_is_B") for i in range(n)}
    raw = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i in range(n):
            if assign[i] == "x_is_A":
                a, b = gens_x[i], gens_y[i]
            else:
                a, b = gens_y[i], gens_x[i]
            futs[ex.submit(judge_pair, qs[i], refs[i], a, b)] = i
        for fut in as_completed(futs):
            raw[futs[fut]] = fut.result()

    per_i = {}
    x_wins = x_losses = ties = errs = 0
    for i in range(n):
        r = raw[i]
        if isinstance(r, dict):
            per_i[i] = "err"
            errs += 1
            continue
        if r == "tie":
            per_i[i] = "tie"
            ties += 1
        else:
            x_pos = "A" if assign[i] == "x_is_A" else "B"
            if r == x_pos:
                per_i[i] = label_x
                x_wins += 1
            else:
                per_i[i] = label_y
                x_losses += 1
    decided = x_wins + x_losses + ties
    return {
        f"{label_x}_wins": x_wins, "ties": ties, f"{label_y}_wins": x_losses,
        "errors": errs, "decided": decided,
        f"{label_x}_win_rate_incl_ties": round(x_wins / decided, 4) if decided else None,
        f"{label_x}_win_rate_excl_ties": round(x_wins / (x_wins + x_losses), 4) if (x_wins + x_losses) else None,
        "note": f"{label_x} placed in a random A/B slot per question to remove position bias.",
    }, per_i, assign, raw


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
    ap.add_argument("--lora_dpo", default="chapter08/runs/dpo_lora_run1",
                    help="LoRA-DPO adapter: accepts a local path, an HF repo id, or "
                         "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch8-dpo-lora)")
    ap.add_argument("--out", default="chapter08/eval/dpo_lora_report.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    gens = {}
    for name, loader in (
        ("base", lambda: load_full(args.base)),
        ("sft", lambda: load_full(args.sft)),
        ("dpo", lambda: load_full(args.dpo)),
        ("lora_dpo", lambda: load_lora(args.sft, args.lora_dpo)),
    ):
        print(f"generating {name}...", flush=True)
        model = loader()
        gens[name] = gen_all(model, tok, prompts)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(gens, open(Path(args.out).with_name("lora_generations_cache.json"), "w"))

    # --- Absolute judge scores for all four models ---
    print("judging absolute scores via gpt-5.5...", flush=True)
    jobs = [(mdl, i) for mdl in gens for i in range(n)]
    judge = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, qs[i], refs[i], gens[mdl][i]): (mdl, i) for mdl, i in jobs}
        for fut in as_completed(futs):
            judge[futs[fut]] = fut.result()

    # --- Pairwise: LoRA-DPO vs SFT, and LoRA-DPO vs full-DPO ---
    print("judging pairwise LoRA-DPO vs SFT...", flush=True)
    pw_sft, perA, _, _ = pairwise(gens["lora_dpo"], gens["sft"], qs, refs,
                                  random.Random(args.seed), n, args.workers, "lora_dpo", "sft")
    print("judging pairwise LoRA-DPO vs full-DPO...", flush=True)
    pw_dpo, perB, _, _ = pairwise(gens["lora_dpo"], gens["dpo"], qs, refs,
                                  random.Random(args.seed + 1), n, args.workers, "lora_dpo", "dpo")

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

    report["pairwise_lora_dpo_vs_sft"] = pw_sft
    report["pairwise_lora_dpo_vs_full_dpo"] = pw_dpo

    per_example = []
    for i in range(n):
        per_example.append({
            "category": cats[i],
            "question": qs[i][:300],
            "reference": refs[i][:400],
            **{f"{mdl}_answer": gens[mdl][i][:800] for mdl in gens},
            **{f"{mdl}_judge": judge[(mdl, i)] for mdl in gens},
            "pw_lora_vs_sft": perA[i],
            "pw_lora_vs_dpo": perB[i],
        })

    out = Path(args.out)
    json.dump(report, open(out, "w"), indent=1)
    write_jsonl(out.with_name("lora_judge_examples.jsonl"), per_example)

    print("\n==== CH8 LoRA-DPO EVAL REPORT ====")
    for mdl in ("base", "sft", "dpo", "lora_dpo"):
        r = report["models"][mdl]
        print(f"{mdl:9s}  tokenF1 {r['token_f1']:.3f} | format {r['format_adherence']*100:4.0f}% | "
              f"correct {r['judge_correctness']} action {r['judge_actionability']} "
              f"format {r['judge_format']} (errs {r['judge_errors']})")
    print(f"\nPairwise LoRA-DPO vs SFT:      "
          f"LoRA-DPO wins {pw_sft['lora_dpo_wins']} / ties {pw_sft['ties']} / SFT wins {pw_sft['sft_wins']}")
    print(f"Pairwise LoRA-DPO vs full-DPO: "
          f"LoRA-DPO wins {pw_dpo['lora_dpo_wins']} / ties {pw_dpo['ties']} / full-DPO wins {pw_dpo['dpo_wins']}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
