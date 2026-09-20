"""Generate responses from the distilled student model.

The student is a LoRA adapter on the base model.  Loading requires
both the base model and the adapter, same as Chapter 5.

Run from code/:
    python -m chapter07.generate \
        --base_model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter_dir chapter07/runs/student_run1 \
        --prompt "How do I troubleshoot a VPN connection failure?"
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.hub import split_ref, subfolder_kwargs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Student model inference")
    ap.add_argument(
        "--base_model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model: accepts a local path, an HF repo id, or 'repo#subfolder' "
        "(e.g. bahree/ModelAdaptationBook#ch6-sft)",
    )
    ap.add_argument(
        "--adapter_dir",
        required=True,
        help="Student LoRA adapter: accepts a local path, an HF repo id, or "
        "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch7-distilled)",
    )
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--system_prompt",
                    default="You are an IT support assistant. Provide clear, step-by-step answers.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    adapter_p, adapter_s = split_ref(args.adapter_dir)
    base_p, base_s = split_ref(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(adapter_p, **subfolder_kwargs(adapter_s), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_p,
        **subfolder_kwargs(base_s),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading student adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_p, **subfolder_kwargs(adapter_s))
    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"\nPrompt:   {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
