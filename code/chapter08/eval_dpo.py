"""Evaluate DPO model: compare base vs SFT vs DPO.

Three-way comparison showing the progressive improvement chain:
base → SFT (Ch6) → DPO (Ch8).

Run from code/:
    python -m chapter08.eval_dpo \
        --data_dir chapter06/data/dolly_sft \
        --sft_dir chapter06/runs/sft_run1 \
        --dpo_dir chapter08/runs/dpo_run1 \
        --output chapter08/eval/dpo_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.jsonl import read_jsonl
from common.seed import seed_everything
from chapter05.metrics import token_f1

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."
MAX_NEW_TOKENS = 256


def generate_response(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_model(model, tokenizer, examples):
    cat_f1 = defaultdict(list)
    for ex in examples:
        msgs = ex["messages"]
        prompt = next(m["content"] for m in msgs if m["role"] == "user")
        reference = next(m["content"] for m in msgs if m["role"] == "assistant")
        category = ex.get("category", "unknown")
        generated = generate_response(model, tokenizer, prompt)
        cat_f1[category].append(token_f1(generated, reference))

    f1_by_cat = {cat: sum(s) / len(s) for cat, s in sorted(cat_f1.items())}
    overall = sum(s for v in cat_f1.values() for s in v) / max(
        sum(len(v) for v in cat_f1.values()), 1
    )
    return f1_by_cat, overall


def load_and_eval(model_path, examples, label, base_model=None):
    """Load model, evaluate, free GPU."""
    print(f"\n--- Evaluating {label} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    f1_by_cat, overall = evaluate_model(model, tokenizer, examples)
    del model
    torch.cuda.empty_cache()
    return f1_by_cat, overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO model")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sft_dir", required=True, help="Ch6 SFT model")
    parser.add_argument("--dpo_dir", required=True, help="Ch8 DPO model")
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_everything(42)
    examples = list(read_jsonl(str(Path(args.data_dir) / "valid.jsonl")))
    print(f"Loaded {len(examples)} test examples")

    # Evaluate all three models
    base_f1, base_overall = load_and_eval(args.base_model, examples, "Base")
    sft_f1, sft_overall = load_and_eval(args.sft_dir, examples, "SFT (Ch6)")
    dpo_f1, dpo_overall = load_and_eval(args.dpo_dir, examples, "DPO (Ch8)")

    # Print comparison
    print("\n" + "=" * 65)
    print(f"{'Category':<25} {'Base':>8} {'SFT':>8} {'DPO':>8}")
    print("-" * 65)
    all_cats = sorted(set(list(base_f1.keys()) + list(sft_f1.keys()) + list(dpo_f1.keys())))
    for cat in all_cats:
        b = base_f1.get(cat, 0.0)
        s = sft_f1.get(cat, 0.0)
        d = dpo_f1.get(cat, 0.0)
        print(f"{cat:<25} {b:>7.3f} {s:>8.3f} {d:>8.3f}")
    print("-" * 65)
    print(f"{'OVERALL':<25} {base_overall:>7.3f} {sft_overall:>8.3f} {dpo_overall:>8.3f}")
    print("=" * 65)

    if args.output:
        report = {
            "num_examples": len(examples),
            "base_overall_f1": base_overall,
            "sft_overall_f1": sft_overall,
            "dpo_overall_f1": dpo_overall,
            "base_f1_by_category": base_f1,
            "sft_f1_by_category": sft_f1,
            "dpo_f1_by_category": dpo_f1,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
