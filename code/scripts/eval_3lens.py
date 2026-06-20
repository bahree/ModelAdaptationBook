"""Three-lens evaluation of base vs LoRA vs SFT on the IT valid set:
  1. token-F1 vs the raw reference  (the naive proxy; expected low on long answers)
  2. format-adherence              (deterministic: does the answer follow the house format)
  3. LLM-as-judge quality          (gpt-5.5 rates correctness / actionability / format)

Generations + per-example judge results are saved so the chapter can show them
without re-running. Run from code/ AFTER training:
    python scripts/eval_3lens.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common.env  # noqa: F401
from common.jsonl import read_jsonl, write_jsonl
from common.openrouter import chat
from chapter05.metrics import token_f1
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
        r = chat([{"role": "user", "content": JUDGE_PROMPT.format(q=q[:1200], ref=ref[:1500], ans=ans[:1800])}],
                 model=JUDGE, max_tokens=300, temperature=0.0)
        m = re.search(r"\{.*\}", r.get("content") or "", re.S)
        d = json.loads(m.group(0))
        return {k: float(d.get(k)) for k in ("correctness", "actionability", "format")}
    except Exception as e:
        return {"correctness": None, "actionability": None, "format": None, "err": repr(e)[:80]}


def load(model_path, adapter=None):
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
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
    ap.add_argument("--lora", default="chapter05/runs/it_lora_fmt")
    ap.add_argument("--sft", default="chapter06/runs/it_sft_fmt")
    ap.add_argument("--out", default="data/it_support_fmt/eval_3lens.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    val = list(read_jsonl(args.valid))
    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = [[{"role": "system", "content": IT_SYSTEM},
                {"role": "user", "content": next(m["content"] for m in ex["messages"] if m["role"] == "user")}]
               for ex in val]
    refs = [next(m["content"] for m in ex["messages"] if m["role"] == "assistant") for ex in val]
    cats = [ex.get("category", "general") for ex in val]
    qs = [p[1]["content"] for p in prompts]

    gens = {}
    print("generating base...", flush=True)
    m = load(BASE)
    gens["base"] = gen_all(m, tok, prompts)
    del m
    torch.cuda.empty_cache()
    print("generating lora...", flush=True)
    m = load(BASE, adapter=args.lora)
    gens["lora"] = gen_all(m, tok, prompts)
    del m
    torch.cuda.empty_cache()
    print("generating sft...", flush=True)
    m = load(args.sft)
    gens["sft"] = gen_all(m, tok, prompts)
    del m
    torch.cuda.empty_cache()

    # judge all (model,example) pairs in parallel
    print("judging via gpt-5.5...", flush=True)
    jobs = [(mdl, i) for mdl in gens for i in range(len(val))]
    judge = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, qs[i], refs[i], gens[mdl][i]): (mdl, i) for mdl, i in jobs}
        for fut in as_completed(futs):
            judge[futs[fut]] = fut.result()

    # aggregate
    report, per_example = {}, []
    for mdl in gens:
        f1s = [token_f1(gens[mdl][i], refs[i]) for i in range(len(val))]
        fmt = [format_ok(gens[mdl][i]) for i in range(len(val))]
        jc = [judge[(mdl, i)].get("correctness") for i in range(len(val))]
        ja = [judge[(mdl, i)].get("actionability") for i in range(len(val))]
        jf = [judge[(mdl, i)].get("format") for i in range(len(val))]
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        report[mdl] = {
            "token_f1": round(sum(f1s) / len(f1s), 4),
            "format_adherence": round(sum(fmt) / len(fmt), 3),
            "judge_correctness": mean(jc),
            "judge_actionability": mean(ja),
            "judge_format": mean(jf),
        }
        # per-category token-F1
        bycat = defaultdict(list)
        for i in range(len(val)):
            bycat[cats[i]].append(f1s[i])
        report[mdl]["token_f1_by_category"] = {c: round(sum(v) / len(v), 4) for c, v in bycat.items()}

    for i in range(len(val)):
        per_example.append({"category": cats[i], "question": qs[i][:300], "reference": refs[i][:400],
                            **{f"{mdl}_answer": gens[mdl][i][:600] for mdl in gens},
                            **{f"{mdl}_judge": judge[(mdl, i)] for mdl in gens}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    write_jsonl(Path(args.out).with_name("eval_3lens_examples.jsonl"), per_example)
    print("\n==== 3-LENS REPORT ====")
    for mdl in ("base", "lora", "sft"):
        r = report[mdl]
        print(f"{mdl:5s}  tokenF1 {r['token_f1']:.3f} | format {r['format_adherence']*100:4.0f}% | "
              f"judge correct {r['judge_correctness']} action {r['judge_actionability']} format {r['judge_format']}")
    print(f"\nSaved {args.out} + examples jsonl")


if __name__ == "__main__":
    main()
