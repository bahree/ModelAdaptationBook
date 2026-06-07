"""Create preference pairs for DPO training.

Generates (prompt, chosen, rejected) triples by using the Ch6 SFT model
as the source of "chosen" responses and the base model as the source of
"rejected" responses.  This is the simplest valid approach to preference
data construction: the fine-tuned model produces better responses than
the base model for our target task.

Run from code/:
    python -m chapter08.scripts.prepare_preference_data \
        --sft_dir chapter06/runs/sft_run1 \
        --prompts chapter06/data/dolly_sft/valid.jsonl \
        --out chapter08/data/preference_pairs \
        --num_prompts 50
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.jsonl import read_jsonl, write_jsonl
from common.manifest import write_json
from common.seed import seed_everything

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
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
        ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    gen_ids = ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def parse_args():
    ap = argparse.ArgumentParser(description="Create DPO preference pairs")
    ap.add_argument("--sft_dir", required=True, help="Path to Ch6 SFT model (chosen source)")
    ap.add_argument("--base_model", default=DEFAULT_MODEL, help="Base model (rejected source)")
    ap.add_argument("--prompts", required=True, help="JSONL with prompts")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--num_prompts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Extract prompts
    examples = list(read_jsonl(args.prompts))[:args.num_prompts]
    prompts = []
    for ex in examples:
        msgs = ex["messages"]
        user_msg = next(m["content"] for m in msgs if m["role"] == "user")
        prompts.append(user_msg)
    print(f"Loaded {len(prompts)} prompts")

    # Generate "chosen" responses from SFT model
    print(f"\nLoading SFT model (chosen source): {args.sft_dir}")
    sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_dir, trust_remote_code=True)
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    sft_model.eval()

    chosen_responses = []
    for i, prompt in enumerate(prompts):
        chosen_responses.append(generate(sft_model, sft_tokenizer, prompt))
        if (i + 1) % 20 == 0:
            print(f"  Chosen: {i + 1}/{len(prompts)}")
    del sft_model
    torch.cuda.empty_cache()

    # Generate "rejected" responses from base model
    print(f"\nLoading base model (rejected source): {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    rejected_responses = []
    for i, prompt in enumerate(prompts):
        rejected_responses.append(generate(base_model, base_tokenizer, prompt))
        if (i + 1) % 20 == 0:
            print(f"  Rejected: {i + 1}/{len(prompts)}")
    del base_model
    torch.cuda.empty_cache()

    # Build preference pairs in DPO format
    pairs = []
    for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
        pairs.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "chosen": [
                {"role": "assistant", "content": chosen},
            ],
            "rejected": [
                {"role": "assistant", "content": rejected},
            ],
        })

    # Split train/valid (80/20)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_pairs)
    write_jsonl(out_dir / "valid.jsonl", valid_pairs)

    manifest = {
        "source": "dpo_preference_pairs",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "chosen_model": args.sft_dir,
        "rejected_model": args.base_model,
        "counts": {"train": len(train_pairs), "valid": len(valid_pairs)},
        "seed": args.seed,
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\nPreference data written to {out_dir}")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Valid: {len(valid_pairs)} pairs")


if __name__ == "__main__":
    main()
