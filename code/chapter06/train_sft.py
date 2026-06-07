"""Full SFT training for Qwen3-4B on chat-formatted JSONL data.

Unlike Chapter 5's LoRA training, this script updates every parameter in
the model.  The base model is NOT frozen.

Run from code/:
    python -m chapter06.train_sft \
        --train chapter06/data/dolly_sft/train.jsonl \
        --valid chapter06/data/dolly_sft/valid.jsonl \
        --out   chapter06/runs/sft_run1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from datasets import Dataset as HFDataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.env import resolve_report_to
from common.jsonl import read_jsonl_list
from common.seed import seed_everything

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full SFT training")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", required=True, help="Training JSONL")
    ap.add_argument("--valid", required=True, help="Validation JSONL")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Fixed step count (useful for smoke tests)")
    ap.add_argument("--report_to", choices=["none", "wandb"], default=None)
    return ap.parse_args()


def load_messages_dataset(jsonl_path: str, system_prompt: str) -> HFDataset:
    """Load JSONL and return an HFDataset with a 'messages' column."""
    rows = read_jsonl_list(jsonl_path)
    messages_rows = []
    for row in rows:
        msgs = row["messages"]
        # Ensure system prompt exists
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        messages_rows.append({"messages": msgs})
    return HFDataset.from_list(messages_rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve W&B ---
    report_to = resolve_report_to(args.report_to)
    report_to_final: List[str] = []
    if report_to == "wandb":
        try:
            import wandb  # noqa: F401
            report_to_final = ["wandb"]
        except ImportError:
            print("W&B requested but not installed; falling back to none.")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Model (full precision, all parameters trainable) ---
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # --- Data ---
    train_ds = load_messages_dataset(args.train, args.system_prompt)
    valid_ds = load_messages_dataset(args.valid, args.system_prompt)
    print(f"Train: {len(train_ds)} examples | Valid: {len(valid_ds)} examples")

    # --- Precision detection ---
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)

    # --- SFTConfig (no peft_config = full parameter training) ---
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=args.max_length,
        report_to=report_to_final if report_to_final else ["none"],
        dataloader_num_workers=2,
    )

    # --- Train ---
    # Note: no peft_config argument.  SFTTrainer will train all parameters.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    print("\n=== Starting full SFT training ===")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate:        {args.lr}")
    print(f"  Epochs:               {args.epochs}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    trainer.train()

    # --- Save ---
    # Full SFT saves the entire model (not just adapters).
    # Expect ~8 GB on disk for a 4B model in bfloat16.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nModel saved to: {out_dir}")


if __name__ == "__main__":
    main()
