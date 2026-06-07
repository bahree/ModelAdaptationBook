"""Generate responses from the DPO-optimized model.

The DPO model is a complete model checkpoint (like Ch6 SFT),
loaded from a single directory.

Run from code/:
    python -m chapter08.generate \
        --model_dir chapter08/runs/dpo_run1 \
        --prompt "How do I troubleshoot a VPN connection failure?"
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def parse_args():
    ap = argparse.ArgumentParser(description="DPO model inference")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Loading DPO model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"\nPrompt:   {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
