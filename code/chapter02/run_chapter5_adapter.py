"""Chapter 2 optional: preview what the same recipe produces at chapter 5 scale.

Loads the LoRA adapter chapter 5 produces (the IT support dataset, three epochs,
properly evaluated) onto the base model and runs the same three demo prompts the
chapter 2 quickstart prints. Each prompt is generated twice, once with the adapter
disabled (base) and once with it enabled, so you can see the qualitative shift.

This is opt-in. Nothing in chapter 2 requires this script. The point is to give
curious readers a preview of what the same recipe produces at chapter 5's full
scale before they read chapter 5 in depth.

Adapter resolution order (first match wins):
    1. Local copy at chapter05/runs/it_lora/ (if you have already run Ch5)
    2. Hugging Face Hub at the published location (cached after first use)
    3. Local chapter 2 quickstart adapter at chapter02/runs/ch2_quickstart/
       (only when --use-quickstart is passed; the quickstart adapter is NOT
       a chapter 5 adapter, the script will say so)

Run from the code/ directory:
    python -m chapter02.run_chapter5_adapter
    python -m chapter02.run_chapter5_adapter --hub-repo bahree/qwen3-4b-it-lora-ch5
    python -m chapter02.run_chapter5_adapter --use-quickstart

If nothing resolves, the script prints the publish/train instructions and exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CH5_LOCAL_PATH = Path("chapter05/runs/it_lora")
QUICKSTART_LOCAL_PATH = Path("chapter02/runs/ch2_quickstart")
DEFAULT_HUB_REPO = "bahree/qwen3-4b-it-lora-ch5"
SYSTEM_PROMPT = "You are a helpful assistant."

# The three prompts the chapter 2 quickstart prints, hardcoded here so this
# script reproduces them without depending on the quickstart's random state.
DEMO_PROMPTS = [
    (
        "Which of the following are deciduous trees?\n\n"
        "Abies concolor\nAcer rubrum\nAcer saccharinum\n"
        "Cornus florida\nQuercus rubra\nPinus strobus"
    ),
    "What are the words of House Wendwater?",
    "What is the difference between love and affection?",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--hub-repo",
        default=DEFAULT_HUB_REPO,
        help=f"Hugging Face Hub repo to fall back to (default: {DEFAULT_HUB_REPO}).",
    )
    ap.add_argument(
        "--use-quickstart",
        action="store_true",
        help=(
            "If neither the local Ch5 adapter nor Hub copy is available, fall back "
            "to the Ch2 quickstart adapter. The quickstart adapter is the 20-step "
            "preview, NOT the chapter 5 production adapter; outputs will not match "
            "what chapter 5 produces."
        ),
    )
    return ap.parse_args()


def find_adapter(args: argparse.Namespace) -> Optional[tuple[str, str]]:
    """Return (human_description, adapter_id) for the first available source."""
    if (CH5_LOCAL_PATH / "adapter_config.json").exists():
        return ("local chapter 5 adapter", str(CH5_LOCAL_PATH))

    try:
        from huggingface_hub import HfApi

        HfApi().repo_info(args.hub_repo)
        return (f"Hugging Face Hub ({args.hub_repo})", args.hub_repo)
    except Exception:
        pass

    if args.use_quickstart and (QUICKSTART_LOCAL_PATH / "adapter_config.json").exists():
        return (
            "local chapter 2 quickstart adapter (preview only, NOT the chapter 5 adapter)",
            str(QUICKSTART_LOCAL_PATH),
        )

    return None


def print_no_adapter_instructions(args: argparse.Namespace) -> None:
    print("Could not find a chapter 5 adapter to load.")
    print()
    print("Checked, in order:")
    print(f"  1. Local path: {CH5_LOCAL_PATH}")
    print(f"  2. Hugging Face Hub: {args.hub_repo}")
    if args.use_quickstart:
        print(f"  3. Local quickstart fallback: {QUICKSTART_LOCAL_PATH}")
    else:
        print("  3. (Local quickstart fallback skipped; pass --use-quickstart to enable)")
    print()
    print("Two ways to fix this:")
    print()
    print("Option A. Train the chapter 5 adapter locally:")
    print("  python scripts/build_it_support_dataset.py")
    print("  python scripts/reformat_it_answers.py")
    print("  python -m chapter05.train_lora \\")
    print("    --train data/it_support_fmt/train.jsonl \\")
    print("    --valid data/it_support/valid.jsonl \\")
    print(f"    --out {CH5_LOCAL_PATH}")
    print()
    print("Option B. Pull the published adapter from Hugging Face Hub once it is up:")
    print(f"  (the script tries {args.hub_repo} automatically when it is published).")
    print()
    print("Option C. Run the chapter 2 quickstart and pass --use-quickstart:")
    print("  python -m chapter02.quickstart")
    print("  python -m chapter02.run_chapter5_adapter --use-quickstart")


def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 120) -> str:
    """Greedy one-shot generation."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    found = find_adapter(args)
    if found is None:
        print_no_adapter_instructions(args)
        sys.exit(1)

    source_desc, adapter_id = found
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    print(f"Attaching adapter from {source_desc}: {adapter_id}")
    model = PeftModel.from_pretrained(base, adapter_id)
    model.eval()

    for i, prompt in enumerate(DEMO_PROMPTS, start=1):
        first_line = prompt.splitlines()[0]
        print(f"\n=== Prompt {i} ===")
        print(f"Q: {first_line[:100]}{'...' if len(first_line) > 100 else ''}")

        with model.disable_adapter():
            base_out = generate(model, tokenizer, prompt)
        print(f"\nBase (no adapter):\n  {base_out[:350]}")

        adapter_out = generate(model, tokenizer, prompt)
        print(f"\nAdapter:\n  {adapter_out[:350]}")


if __name__ == "__main__":
    main()
