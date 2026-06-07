"""Generate training data from a teacher model for knowledge distillation.

The teacher is the Chapter 6 SFT model.  We run it on a set of diverse
prompts and collect high-quality responses that the student will learn
to reproduce.

Run from code/:
    python -m chapter07.scripts.generate_teacher_data \
        --teacher_dir chapter06/runs/sft_run1 \
        --prompts chapter06/data/dolly_sft/train.jsonl \
        --out chapter07/data/teacher_outputs.jsonl \
        --num_prompts 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.jsonl import read_jsonl, write_jsonl
from common.seed import seed_everything


DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate teacher data for distillation")
    ap.add_argument("--teacher_dir", required=True, help="Path to teacher model (Ch6 SFT)")
    ap.add_argument("--prompts", required=True, help="JSONL with prompts to run through teacher")
    ap.add_argument("--out", required=True, help="Output JSONL for teacher responses")
    ap.add_argument("--num_prompts", type=int, default=200, help="Number of prompts to use")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def generate_response(model, tokenizer, prompt: str, system_prompt: str,
                      max_new_tokens: int, temperature: float) -> str:
    """Generate a single response from the teacher model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Load prompts from Ch6 training data
    all_examples = list(read_jsonl(args.prompts))
    prompts = []
    for ex in all_examples[:args.num_prompts]:
        msgs = ex["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        category = ex.get("category", "unknown")
        prompts.append({"prompt": user_msg, "category": category})

    print(f"Loaded {len(prompts)} prompts")

    # Load teacher model
    print(f"Loading teacher model from {args.teacher_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Generate teacher responses
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, item in enumerate(prompts):
        response = generate_response(
            model, tokenizer, item["prompt"], DEFAULT_SYSTEM_PROMPT,
            args.max_new_tokens, args.temperature,
        )
        results.append({
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "category": item["category"],
            "source": "teacher",
        })
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} responses")

    write_jsonl(out_path, results)
    print(f"\nTeacher data written to {out_path}")
    print(f"  Total examples: {len(results)}")


if __name__ == "__main__":
    main()
