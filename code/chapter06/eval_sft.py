"""Evaluate base vs. fine-tuned SFT model on held-out test data.

Uses token_f1, exact_match, and is_refusal from chapter05.metrics
for consistency with Chapter 5's evaluation.

Run from code/:
    python -m chapter06.eval_sft \
        --data_dir chapter06/data/dolly_sft \
        --model_dir chapter06/runs/sft_run1 \
        --output chapter06/runs/sft_run1/eval_report.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.jsonl import read_jsonl
from common.seed import seed_everything
from chapter05.metrics import token_f1, exact_match, is_refusal

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."
MAX_NEW_TOKENS = 256


def load_model_and_tokenizer(model_path: str,
                             device: str = "auto"):
    """Load a model and its tokenizer.

    Works for both HuggingFace model IDs (base model)
    and local checkpoint directories (fine-tuned model).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str,
                      system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Generate a single response using chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def extract_prompt_and_response(example: dict) -> tuple:
    """Extract user prompt, assistant response, and category from
    a chat-formatted JSONL example."""
    msgs = example["messages"]
    prompt = next(m["content"] for m in msgs if m["role"] == "user")
    response = next(m["content"] for m in msgs if m["role"] == "assistant")
    category = example.get("category", "unknown")
    return prompt, response, category


def evaluate_model(model, tokenizer, examples):
    """Run evaluation and return per-category token-F1, exact match,
    and safety refusal rate."""
    cat_f1 = defaultdict(list)
    cat_em = defaultdict(list)
    refusal_flags = []
    results = []

    for ex in examples:
        prompt, reference, category = extract_prompt_and_response(ex)

        generated = generate_response(model, tokenizer, prompt)

        f1 = token_f1(generated, reference)
        em = exact_match(generated, reference)
        refused = is_refusal(generated)

        cat_f1[category].append(f1)
        cat_em[category].append(int(em))
        refusal_flags.append(refused)
        results.append({
            "prompt": prompt[:80],
            "category": category,
            "token_f1": round(f1, 3),
            "exact_match": em,
            "is_refusal": refused,
            "generated": generated[:200],
            "reference": reference[:200],
        })

    f1_by_category = {
        cat: sum(scores) / len(scores)
        for cat, scores in sorted(cat_f1.items())
    }
    overall_f1 = (
        sum(s for scores in cat_f1.values() for s in scores)
        / max(sum(len(s) for s in cat_f1.values()), 1)
    )
    refusal_rate = sum(refusal_flags) / max(len(refusal_flags), 1)
    return f1_by_category, overall_f1, refusal_rate, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs. fine-tuned SFT model"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing valid.jsonl test split"
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--base_model", default=BASE_MODEL,
        help="Base model ID for comparison"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save evaluation report (JSON)"
    )
    args = parser.parse_args()

    seed_everything(42)

    # Load test examples
    valid_path = Path(args.data_dir) / "valid.jsonl"
    examples = list(read_jsonl(str(valid_path)))
    print(f"Loaded {len(examples)} test examples from {valid_path}")

    # Evaluate base model
    print("\n--- Evaluating base model ---")
    base_model, base_tok = load_model_and_tokenizer(args.base_model)
    base_f1, base_overall, base_refusal, _ = evaluate_model(
        base_model, base_tok, examples
    )
    del base_model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    print("\n--- Evaluating fine-tuned model ---")
    ft_model, ft_tok = load_model_and_tokenizer(args.model_dir)
    ft_f1, ft_overall, ft_refusal, ft_results = evaluate_model(
        ft_model, ft_tok, examples
    )

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Category':<25} {'Base F1':>10} {'FT F1':>10} {'Delta':>8}")
    print("-" * 65)
    all_cats = sorted(
        set(list(base_f1.keys()) + list(ft_f1.keys()))
    )
    for cat in all_cats:
        b = base_f1.get(cat, 0.0)
        f = ft_f1.get(cat, 0.0)
        delta = f - b
        sign = "+" if delta >= 0 else ""
        print(f"{cat:<25} {b:>9.3f} {f:>9.3f} {sign}{delta:>7.3f}")
    print("-" * 65)
    delta_overall = ft_overall - base_overall
    sign = "+" if delta_overall >= 0 else ""
    print(
        f"{'OVERALL TOKEN-F1':<25} {base_overall:>9.3f}"
        f" {ft_overall:>9.3f} {sign}{delta_overall:>7.3f}"
    )
    print(
        f"{'SAFETY REFUSAL RATE':<25} {base_refusal:>9.0%}"
        f" {ft_refusal:>9.0%}"
    )
    print("=" * 65)

    # Save report
    if args.output:
        report = {
            "base_model": args.base_model,
            "fine_tuned_model": args.model_dir,
            "num_examples": len(examples),
            "base_f1_by_category": base_f1,
            "ft_f1_by_category": ft_f1,
            "base_overall_f1": base_overall,
            "ft_overall_f1": ft_overall,
            "base_refusal_rate": base_refusal,
            "ft_refusal_rate": ft_refusal,
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
