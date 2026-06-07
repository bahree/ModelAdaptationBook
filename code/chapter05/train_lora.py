"""LoRA fine-tuning script using TRL's SFTTrainer (Listing 5.2).

Trains a LoRA adapter on chat-formatted JSONL data and saves the adapter
weights. The base model is frozen; only the small LoRA matrices are updated.

Usage:
    python -m chapter05.train_lora \\
        --train chapter05/data/dolly_subset/train.jsonl \\
        --valid chapter05/data/dolly_subset/valid.jsonl \\
        --out chapter05/runs/dolly_lora

See Chapter 5, Section 5.1 (Step 2) and the README for full details.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from trl import SFTConfig, SFTTrainer

from chapter05 import DEFAULT_MODEL_NAME
from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT
from chapter05.data import load_chat_jsonl
from chapter05.dataset import prepare_dataset_for_sft
from chapter05.modeling import create_lora_config, load_base_model_lora, load_tokenizer
from common.env import resolve_report_to
from common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LoRA training.

    Returns:
        Namespace with training configuration (model, data paths,
        hyperparameters, LoRA settings, and logging options).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--train", required=True, help="Training JSONL (messages or prompt/response)")
    ap.add_argument("--valid", required=True, help="Validation JSONL (messages or prompt/response)")
    ap.add_argument("--out", required=True, help="Output directory for adapter")

    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)

    # Training hyperparameters
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    # 2e-4 is the standard LoRA learning rate -- higher than full SFT because
    # we update far fewer parameters, so each update can be larger.
    ap.add_argument("--lr", type=float, default=2e-4)
    # batch_size=1 with grad_accum=8 gives an effective batch size of 8,
    # balancing memory usage (fits on 8-12 GB GPUs) with training stability.
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # LoRA hyperparameters (see Section 5.5 for guidance on choosing these)
    # r=16 is the sweet spot for most tasks (Section 5.5, Table 5.4).
    # alpha=2*r is the standard convention for the scaling factor.
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging and checkpointing
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=0)  # reserved for future use
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set > 0, run for a fixed number of steps (useful for smoke tests).",
    )

    ap.add_argument("--report_to", choices=["none", "wandb"], default=None)
    return ap.parse_args()


def main() -> None:
    """Load base model, attach LoRA, train on JSONL data, and save the adapter."""
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Weights & Biases is opt-in; gracefully fall back if not installed.
    report_to = resolve_report_to(args.report_to)
    report_to_final: List[str] = []
    if report_to != "none":
        if report_to == "wandb":
            try:
                import wandb  # noqa: F401
            except Exception:
                print("W&B enabled but not installed; falling back to report_to=none.")
            else:
                report_to_final = ["wandb"]

    tokenizer = load_tokenizer(args.model)
    model = load_base_model_lora(args.model, gradient_checkpointing=True)
    
    # Create LoRA config -- SFTTrainer will apply it to the model automatically
    # via the peft_config parameter, so we don't call get_peft_model() ourselves.
    lora_config = create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Load data and convert to the HF Dataset format SFTTrainer expects.
    train_ex = load_chat_jsonl(args.train, system_prompt=args.system_prompt)
    valid_ex = load_chat_jsonl(args.valid, system_prompt=args.system_prompt)
    train_ds = prepare_dataset_for_sft(train_ex)
    valid_ds = prepare_dataset_for_sft(valid_ex)

    # Auto-detect best precision: prefer bf16 (better dynamic range) over fp16.
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16 = bool(use_cuda and not use_bf16)

    targs = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",       # TRL 0.9+ renamed evaluation_strategy to eval_strategy
        eval_steps=args.eval_steps,
        save_strategy="no",          # We save the final adapter at the end, not intermediate checkpoints
        report_to=report_to_final or "none",
        fp16=use_fp16,
        bf16=use_bf16,
        max_length=args.max_length,  # TRL 0.9+ moved max_seq_length into SFTConfig as max_length
    )

    # SFTTrainer handles LoRA application, dataset tokenization, and
    # loss computation (on assistant tokens only when using chat format).
    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora_config,     # SFTTrainer applies LoRA adapters from this config
        processing_class=tokenizer,
    )
    trainer.train()

    # Save only the adapter weights (small, ~tens of MB) and the tokenizer.
    # The base model is NOT duplicated -- load it separately at inference time.
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved LoRA adapter to: {out_dir}")


if __name__ == "__main__":
    main()

