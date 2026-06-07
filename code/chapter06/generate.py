"""Generate responses from a fully fine-tuned SFT model.

Unlike Chapter 5's generate.py (which loads base + adapter), this script
loads the complete model from a single directory.

Run from code/:
    python -m chapter06.generate \
        --model_dir chapter06/runs/sft_run1 \
        --prompt "How do I troubleshoot a VPN connection failure?"
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT model inference")
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned model")
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--system_prompt", default="You are an IT support assistant. Provide clear, step-by-step answers.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"\nPrompt:   {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
