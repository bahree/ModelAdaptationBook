"""Inference script: generate text with the base model and an optional LoRA/QLoRA adapter (Listing 5.4).

Loads the base model, optionally attaches a LoRA or QLoRA adapter, and generates
a response for a single user prompt. Supports adapter merging for deployment.

Usage (base model only):
    python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 \\
        --prompt "Explain how photosynthesis works in simple terms."

Usage (with LoRA adapter):
    python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter chapter05/runs/dolly_lora \\
        --prompt "Explain how photosynthesis works in simple terms."

Usage (with QLoRA adapter -- must use --quantized_4bit):
    python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter chapter05/runs/dolly_qlora --quantized_4bit \\
        --prompt "Explain how photosynthesis works in simple terms."

See Chapter 5, Section 5.1 (Step 4) and the README for full details.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from chapter05 import DEFAULT_MODEL_NAME
from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT, build_prompt_text
from chapter05.modeling import load_base_model_lora, load_base_model_qlora, load_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Namespace with base model, adapter path, prompt, generation settings,
        and optional merge/quantization flags.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--adapter", default=None, help="Path to LoRA/QLoRA adapter folder")
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--quantized_4bit", action="store_true", help="Load base in 4-bit (requires bitsandbytes)")
    ap.add_argument("--merge_adapter", action="store_true", help="Merge adapter into base before generation")
    ap.add_argument("--save_merged", default=None, help="If set, save merged model to this folder")
    return ap.parse_args()


def main() -> None:
    """Load model, optionally attach adapter, and generate a response."""
    args = parse_args()
    tokenizer = load_tokenizer(args.base)

    # Use --quantized_4bit when running a QLoRA-trained adapter so the base
    # model is loaded in 4-bit (matching the precision used during training).
    if args.quantized_4bit:
        model = load_base_model_qlora(args.base, gradient_checkpointing=False)
    else:
        model = load_base_model_lora(args.base, gradient_checkpointing=False)

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge_adapter:
            # merge_and_unload() permanently folds LoRA weights into the base.
            # This loses modularity (can't swap adapters) but can be faster
            # for high-throughput serving. See Section 5.11 deployment options.
            model = model.merge_and_unload()
            if args.save_merged:
                Path(args.save_merged).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.save_merged)
                tokenizer.save_pretrained(args.save_merged)

    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    text = build_prompt_text(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            # Pass temperature=None when not sampling to avoid HF warnings
            # about unused generation parameters.
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # skip_special_tokens=False to show the full chat template (system/user/assistant
    # markers). This is useful for debugging and demonstrating the template structure.
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    print(decoded)


if __name__ == "__main__":
    main()

