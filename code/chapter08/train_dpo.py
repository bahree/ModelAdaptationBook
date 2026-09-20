"""DPO training: preference optimization on the Ch6 SFT checkpoint.

Uses TRL's DPOTrainer to train the model to prefer "chosen" responses
over "rejected" responses, without an explicit reward model.

Run from code/:
    python -m chapter08.train_dpo \
        --model_dir chapter06/runs/sft_run1 \
        --train chapter08/data/preference_pairs/train.jsonl \
        --valid chapter08/data/preference_pairs/valid.jsonl \
        --out   chapter08/runs/dpo_run1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from datasets import Dataset as HFDataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.env import resolve_report_to
from common.gpu import report_peak_gpu_memory
from common.jsonl import read_jsonl_list
from common.seed import seed_everything

DEFAULT_MODEL_DIR = "chapter06/runs/sft_run1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DPO preference training")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                    help="Starting model (Ch6 SFT checkpoint)")
    ap.add_argument("--train", required=True, help="Training preference JSONL")
    ap.add_argument("--valid", required=True, help="Validation preference JSONL")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO beta parameter (KL penalty strength)")
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="Learning rate (lower than SFT)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--report_to", choices=["none", "wandb"], default=None)

    # LoRA-DPO path: train an adapter instead of full weights. With a peft_config,
    # TRL's DPOTrainer uses the adapter-disabled base as the reference policy, so
    # no separate frozen reference copy is held in memory -- this is what makes
    # preference tuning fit on a single 24 GB card.
    ap.add_argument("--lora", action="store_true",
                    help="Train a LoRA adapter (peft) instead of full weights")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()


# Match the LoRA target modules used by chapter05/chapter06 SFT (Qwen3 attn + MLP).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_preference_dataset(jsonl_path: str) -> HFDataset:
    """Load preference pairs into HFDataset format for DPOTrainer."""
    rows = read_jsonl_list(jsonl_path)
    return HFDataset.from_list(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_to = resolve_report_to(args.report_to)
    report_to_final: List[str] = []
    if report_to == "wandb":
        try:
            import wandb  # noqa: F401
            report_to_final = ["wandb"]
        except ImportError:
            print("W&B requested but not installed; falling back to none.")

    # Load tokenizer and model from SFT checkpoint
    print(f"Loading SFT model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load preference data
    train_ds = load_preference_dataset(args.train)
    valid_ds = load_preference_dataset(args.valid)
    print(f"Train: {len(train_ds)} pairs | Valid: {len(valid_ds)} pairs")

    # Precision detection
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())

    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        report_to=report_to_final if report_to_final else ["none"],
        gradient_checkpointing=True,
    )

    # Optional LoRA adapter config. Passing peft_config makes DPOTrainer wrap the
    # model with PEFT and use the adapter-disabled base as the reference policy.
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n=== Starting DPO training ===")
    print(f"  Mode:              {'LoRA-DPO (adapter)' if args.lora else 'full-parameter DPO'}")
    print(f"  Beta (KL penalty): {args.beta}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Starting from:     {args.model_dir}")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.train()

    # DPO holds the policy plus a frozen reference copy, so full-parameter DPO needs
    # more than full SFT; --lora fits one 24 GB card.
    report_peak_gpu_memory("DPO training")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nDPO model saved to: {out_dir}")


if __name__ == "__main__":
    main()
