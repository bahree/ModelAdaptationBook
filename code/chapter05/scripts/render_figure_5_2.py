#!/usr/bin/env python3
"""Render Figure 5.2 (base model vs. LoRA-adapted inference).

Produces a clean print-friendly figure with white background and dark text,
showing the same prompt run through:

  Left panel:  Base Qwen3-4B-Instruct-2507
  Right panel: Base + the Chapter 5 LoRA adapter

Saves both PNG (for the docx) and SVG (for the graphics team to edit).

Rebecca's feedback on the v0.12 docx was that the original figure was a
terminal screenshot with a dark background and light text -- hard to read
on-screen and harder to read in print. This script generates a clean
matplotlib comparison that reproduces deterministically from greedy decoding.

Usage (from the repo's `code/` directory with venv activated):

    python -m chapter05.scripts.render_figure_5_2

Or with a local adapter or different prompt:

    python -m chapter05.scripts.render_figure_5_2 \\
        --adapter chapter05/runs/it_lora \\
        --prompt "Explain how photosynthesis works in simple terms."

Optional flags:

    --prompt          User prompt (default: photosynthesis question)
    --adapter         HF Hub id or local path (default: bahree/qwen3-4b-it-lora-ch5)
    --max_new_tokens  Generation length cap (default: 220)
    --seed            Torch seed for reproducibility (default: 42)
    --out_dir         Where to save the figure (default: ../docs/chapter5/images,
                      i.e., the repo's docs/chapter5/images/ when run from code/)
    --out_name        Filename stem (default: figure_5_2_base_vs_lora)

Outputs land at:

    <out_dir>/<out_name>.png   (for the docx)
    <out_dir>/<out_name>.svg   (for the graphics team)
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Matplotlib is imported lazily so users get a cleaner error if it's missing
try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("matplotlib not installed. Install with: pip install matplotlib")

try:
    from peft import PeftModel
except ImportError:
    sys.exit("peft not installed. Install with: pip install peft")


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PROMPT = "Explain how photosynthesis works in simple terms."
DEFAULT_ADAPTER = "bahree/qwen3-4b-it-lora-ch5"
SYSTEM = "You are a helpful assistant."


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Run greedy generation against `model` and return the model's reply."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def render(prompt: str, base_text: str, lora_text: str, out_dir: Path, out_name: str) -> None:
    """Render the side-by-side comparison figure and save as PNG + SVG."""
    # Keep text as text in the SVG so the graphics team can edit/restyle.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(1, 2, figsize=(14, 9), facecolor="white")

    panels = [
        ("Base model (Qwen3-4B-Instruct-2507)", base_text),
        ("Base model + LoRA adapter", lora_text),
    ]
    for ax, (title, text) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.8)

        # Header bar
        header_height = 0.07
        ax.add_patch(
            mpatches.Rectangle(
                (0, 1 - header_height), 1, header_height,
                transform=ax.transAxes,
                facecolor="#E8E8E8",
                edgecolor="none",
            )
        )
        ax.text(
            0.02, 1 - header_height / 2,
            title,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            va="center", ha="left",
            color="#1A1A1A",
        )

        # Prompt sub-header
        ax.text(
            0.02, 1 - header_height - 0.03,
            f"Prompt: {prompt}",
            transform=ax.transAxes,
            fontsize=10, style="italic",
            color="#555555",
            va="top", ha="left",
        )

        # Output, monospace, wrapped
        wrapped = textwrap.fill(text, width=58, replace_whitespace=False, drop_whitespace=False)
        ax.text(
            0.02, 1 - header_height - 0.10,
            wrapped,
            transform=ax.transAxes,
            family="monospace",
            fontsize=9.5,
            color="#1A1A1A",
            va="top", ha="left",
        )

    fig.suptitle(
        "Figure 5.2: Base model vs. LoRA-adapted inference for the same prompt",
        fontsize=12, fontweight="bold", y=0.98, color="#1A1A1A",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_name}.png"
    svg_path = out_dir / f"{out_name}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  PNG: {png_path} ({png_path.stat().st_size:,} bytes)")
    print(f"  SVG: {svg_path} ({svg_path.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="HF Hub id (default) or local adapter directory")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="../docs/chapter5/images")
    parser.add_argument("--out_name", default="figure_5_2_base_vs_lora")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter:    {args.adapter}")
    print(f"Prompt:     {args.prompt!r}")
    print()

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"  Loaded on {next(base_model.parameters()).device}")

    print()
    print("Generating with base model...")
    base_text = generate(base_model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"  Output ({len(base_text)} chars)")

    print()
    print("Loading LoRA adapter on top of base model...")
    lora_model = PeftModel.from_pretrained(base_model, args.adapter)
    print("  Adapter loaded")

    print()
    print("Generating with LoRA-adapted model...")
    lora_text = generate(lora_model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"  Output ({len(lora_text)} chars)")

    print()
    print("Rendering figure...")
    render(args.prompt, base_text, lora_text, Path(args.out_dir), args.out_name)

    print()
    print("Done. Replace the old image3.png in the docx with the new PNG.")
    print("Graphics team can edit the SVG to match Manning's house style.")


if __name__ == "__main__":
    main()
