"""Step 1 of the hands-on project: prepare the dataset.

SUPERSEDED: the book now trains on the IT support dataset, which is built by
the shared builder ``scripts/build_it_support_dataset.py`` (Stack Exchange IT
Q&A plus a small Dolly mix-in for general-capability retention) and put into
house format by ``scripts/reformat_it_answers.py``. Prepare the data with:

  python scripts/build_it_support_dataset.py     # writes data/it_support/
  python scripts/reformat_it_answers.py           # writes data/it_support_fmt/

Then train/evaluate against:
  --train data/it_support_fmt/train.jsonl
  --valid data/it_support/valid.jsonl

This module is retained only because earlier drafts referenced it and the
``dolly_to_messages`` helper below documents the original Dolly format. Running
it no longer downloads Dolly; it prints the commands above and exits.

Run from the repo root (code/) so that chapter05 and common are importable.
"""
from __future__ import annotations

import argparse

from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT


def dolly_to_messages(
    instruction: str,
    context: str | None,
    response: str,
    *,
    system_prompt: str,
) -> dict:
    """Convert Dolly format (instruction, context, response) to messages format.
    
    Dolly format:
    - instruction: The task/question
    - context: Optional background information
    - response: The answer/output
    
    We combine instruction + context (if present) into the user message.
    """
    # Combine instruction and context for user message
    if context and context.strip():
        user_content = f"{context}\n\n{instruction}"
    else:
        user_content = instruction
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (kept for backward compatibility)."""
    ap = argparse.ArgumentParser(
        description=(
            "Superseded. The book trains on the IT support dataset; build it with "
            "scripts/build_it_support_dataset.py and scripts/reformat_it_answers.py."
        )
    )
    ap.add_argument(
        "--out",
        default="data/it_support",
        help="Where the shared builder writes the dataset (default: data/it_support)",
    )
    ap.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (informational; the shared builder sets this).",
    )
    return ap.parse_args()


def main() -> None:
    """Point the reader at the shared IT support dataset builder and exit."""
    args = parse_args()
    print("Step 1: Prepare the dataset")
    print()
    print("This script is superseded. The book trains on the IT support dataset.")
    print("Build it with the shared builder, then put answers in house format:")
    print()
    print("  python scripts/build_it_support_dataset.py     # writes data/it_support/")
    print("  python scripts/reformat_it_answers.py           # writes data/it_support_fmt/")
    print()
    print(f"Expected output folder: {args.out}")
    print()
    print("Then train and evaluate against:")
    print("  --train data/it_support_fmt/train.jsonl")
    print("  --valid data/it_support/valid.jsonl")


if __name__ == "__main__":
    main()
