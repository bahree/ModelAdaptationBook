"""Demo: run two LoRA adapters on a single base model using PEFT adapter switching.

Shows how to load one base model, attach multiple named adapters, and switch
between them at inference time without reloading the base. This is the approach
discussed in Section 5.8 (Multi-LoRA).

Usage:
    python -m chapter05.multi_lora_demo \\
        --adapter_a chapter05/runs/it_lora \\
        --adapter_b chapter05/runs/it_lora_r8 \\
        --prompt "Explain how photosynthesis works in simple terms."
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel

from chapter05 import DEFAULT_MODEL_NAME
from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT, build_prompt_text
from chapter05.modeling import load_base_model_lora, load_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the multi-adapter demo.

    Returns:
        Namespace with base model, two adapter paths, prompt, and generation settings.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--adapter_a", required=True, help="Adapter A folder path")
    ap.add_argument("--adapter_b", required=True, help="Adapter B folder path")
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    return ap.parse_args()


def generate(model, tokenizer, messages, *, max_new_tokens: int) -> str:
    """Generate a response from the currently active adapter using greedy decoding."""
    text = build_prompt_text(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    """Load base model, attach two adapters, and compare their outputs side-by-side."""
    args = parse_args()

    tokenizer = load_tokenizer(args.base)
    base = load_base_model_lora(args.base, gradient_checkpointing=False)

    # Load adapter A first (this creates the PeftModel), then attach adapter B.
    # Both adapters share the same frozen base weights in GPU memory.
    model = PeftModel.from_pretrained(base, args.adapter_a, adapter_name="adapter_a")
    model.load_adapter(args.adapter_b, adapter_name="adapter_b")

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]

    # Switch between adapters using set_adapter() -- no model reload needed.
    model.set_adapter("adapter_a")
    a = generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

    model.set_adapter("adapter_b")
    b = generate(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)

    print("=== Adapter A ===")
    print(a)
    print("")
    print("=== Adapter B ===")
    print(b)


if __name__ == "__main__":
    main()

