#!/usr/bin/env python3
"""
Reproduces the §1.6 sidebar in Chapter 1 ("What the gap actually looks like").

Runs the same prompt ("I forgot my Outlook password. What should I do?")
through three configurations of Qwen3-4B-Instruct-2507:

  1. The base model (always runs)
  2. The base + the LoRA adapter from Chapter 5 (runs if adapter present)
  3. The full SFT model from Chapter 6 (runs if model present)

The chapter sidebar uses the actual outputs from this script on an NVIDIA
A30 with greedy decoding. If you do not yet have the Chapter 5 / Chapter 6
artifacts on disk (run those chapters first to produce them), the script
runs the base model unconditionally and prints a note for the missing
configurations so you can compare against the chapter sidebar.

Usage (from the repo's `code/` directory with venv activated):

    python -m chapter01.run_sidebar_example

Or run as a script directly:

    python code/chapter01/run_sidebar_example.py

Optional flags:

    --lora_dir   path to a LoRA adapter directory (default: chapter05/runs/it_lora)
    --sft_dir    path to a full SFT model directory (default: chapter06/runs/sft_run1)
    --output     where to save the JSON outputs (default: chapter01/sidebar_outputs.json)
    --base_only  skip LoRA and SFT runs even if their directories exist
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: PEFT is only needed if a LoRA adapter is present
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


PROMPT = "I forgot my Outlook password. What should I do?"
SYSTEM = "You are an IT and HR support assistant for an enterprise help desk."

# Note: an earlier version of this script used a fictional-policy prompt
# ("policy RE-227") to demonstrate confabulation. That prompt produced
# nearly-identical LoRA and SFT outputs (differed by one word) so the
# sidebar could not show the adaptation-confidence ratchet the chapter
# argues for. The current prompt about an Outlook password reset shows
# three visibly different outputs: base hedges correctly, LoRA strips
# the hedge but stays correct, full SFT writes a confidently-wrong
# 10-step procedure that invents UI steps Outlook does not have.
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

GEN_KWARGS = dict(
    max_new_tokens=260,       # base needs ~200 tokens for the full hedged reply
    do_sample=False,          # greedy, deterministic
    repetition_penalty=1.05,
)


def chat_prompt(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user",   "content": user}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt_text: str, label: str) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"\n=== {label} ===\n{text}")
    return text


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo_code = Path(__file__).resolve().parent.parent
    parser.add_argument("--lora_dir", type=Path,
                        default=repo_code / "chapter05" / "runs" / "it_lora",
                        help="Path to a LoRA adapter directory.")
    parser.add_argument("--sft_dir", type=Path,
                        default=repo_code / "chapter06" / "runs" / "sft_run1",
                        help="Path to a full SFT model directory.")
    parser.add_argument("--output", type=Path,
                        default=repo_code / "chapter01" / "sidebar_outputs.json",
                        help="Where to save the JSON of all collected outputs.")
    parser.add_argument("--base_only", action="store_true",
                        help="Run only the base model; skip LoRA and SFT.")
    args = parser.parse_args()

    results = {"prompt": PROMPT, "system": SYSTEM,
               "model": BASE_MODEL, "decoding": "greedy"}

    # --- Base ---
    print(f"[load] base model {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    pt = chat_prompt(tok, SYSTEM, PROMPT)
    results["base"] = generate(base, tok, pt, "BASE Qwen3-4B-Instruct-2507")

    # --- Base + LoRA (optional) ---
    lora_present = (args.lora_dir.exists()
                    and (args.lora_dir / "adapter_config.json").exists())
    if args.base_only:
        print("\n[skip] --base_only set; skipping LoRA run.")
        results["lora"] = None
    elif not HAS_PEFT:
        print("\n[skip] `peft` not installed; cannot load LoRA adapter.")
        print("       pip install peft to enable this configuration.")
        results["lora"] = None
    elif not lora_present:
        print(f"\n[skip] LoRA adapter not found at {args.lora_dir}.")
        print( "       Run Chapter 5 first to produce it_lora, or pass --lora_dir.")
        results["lora"] = None
    else:
        print(f"\n[load] LoRA adapter on top of base: {args.lora_dir}")
        base_for_lora = PeftModel.from_pretrained(base, str(args.lora_dir))
        results["lora"] = generate(base_for_lora, tok, pt,
                                   f"BASE + LoRA adapter ({args.lora_dir.name})")
        del base_for_lora

    # Free the base model before loading the SFT model (separate full set of weights)
    del base
    free()

    # --- Full SFT (optional) ---
    sft_present = (args.sft_dir.exists()
                   and (args.sft_dir / "config.json").exists())
    if args.base_only:
        print("\n[skip] --base_only set; skipping SFT run.")
        results["sft"] = None
    elif not sft_present:
        print(f"\n[skip] SFT model not found at {args.sft_dir}.")
        print( "       Run Chapter 6 first to produce sft_run1, or pass --sft_dir.")
        results["sft"] = None
    else:
        print(f"\n[load] Full SFT model: {args.sft_dir}")
        # The tokenizer files saved alongside the SFT checkpoint may have a
        # `special_tokens_map.json` format that newer transformers versions
        # reject (a list where a dict is expected). The tokens themselves
        # are identical to the base model, so load the tokenizer from the
        # base model id to dodge the incompatibility.
        sft_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        sft = AutoModelForCausalLM.from_pretrained(
            str(args.sft_dir), torch_dtype=torch.bfloat16, device_map="auto"
        )
        pt2 = chat_prompt(sft_tok, SYSTEM, PROMPT)
        results["sft"] = generate(sft, sft_tok, pt2,
                                  f"Full SFT model ({args.sft_dir.name})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
