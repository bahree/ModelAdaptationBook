"""QLoRA fine-tuning script using TRL's SFTTrainer (Listing 5.5).

Same pipeline as train_lora.py but loads the base model in 4-bit quantization
(NF4 via bitsandbytes), reducing GPU memory by roughly 4x. This enables
fine-tuning on GPUs with as little as 6 GB VRAM.

Key differences from LoRA (train_lora.py):
    - Base model loaded with load_base_model_qlora (4-bit NF4 + double quantization)
    - Default rank r=8 (vs r=16 for LoRA) -- the quantized base has less capacity
      to leverage high-rank adapters, and lower rank saves additional memory.
    - Training is ~30-50% slower due to quantization/dequantization overhead.

Usage:
    python -m chapter05.train_qlora \\
        --train chapter05/data/dolly_subset/train.jsonl \\
        --valid chapter05/data/dolly_subset/valid.jsonl \\
        --out chapter05/runs/dolly_qlora

See Chapter 5, Section 5.7 (QLoRA) and Section 5.1 (Step 5) for details.
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
from chapter05.modeling import create_lora_config, load_base_model_qlora, load_tokenizer
from common.env import resolve_report_to
from common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for QLoRA training.

    Returns:
        Namespace with training configuration. Defaults differ from LoRA:
        r=8 and alpha=16 (vs r=16 and alpha=32 for LoRA).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--train", required=True, help="Training JSONL (messages or prompt/response)")
    ap.add_argument("--valid", required=True, help="Validation JSONL (messages or prompt/response)")
    ap.add_argument("--out", required=True, help="Output directory for adapter")

    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)

    # Training hyperparameters (same as LoRA)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # QLoRA uses lower rank than LoRA by default: r=8 (vs r=16) because the
    # 4-bit quantized base has less capacity to leverage high-rank adapters.
    # alpha=2*r keeps the standard scaling convention.
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set > 0, run for a fixed number of steps (useful for smoke tests).",
    )

    ap.add_argument("--report_to", choices=["none", "wandb"], default=None)
    return ap.parse_args()


def main() -> None:
    """Load base model in 4-bit, attach LoRA, train, and save the QLoRA adapter."""
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

    # load_base_model_qlora uses BitsAndBytesConfig with NF4 quantization and
    # double quantization, then calls prepare_model_for_kbit_training to enable
    # gradient checkpointing on quantized weights. See modeling.py for details.
    model = load_base_model_qlora(
        args.model,
        compute_dtype=torch.bfloat16,
        gradient_checkpointing=True,
    )
    
    # LoRA config is the same structure as for full-precision LoRA; only the
    # defaults differ (r=8 vs r=16). SFTTrainer applies it automatically.
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
        save_strategy="no",          # Save final adapter at the end, not intermediate checkpoints
        report_to=report_to_final or "none",
        fp16=use_fp16,
        bf16=use_bf16,
        max_length=args.max_length,  # TRL 0.9+ moved max_seq_length into SFTConfig as max_length
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save adapter weights and tokenizer. The base model is NOT saved here --
    # at inference time, load the base separately with --quantized_4bit and
    # attach this adapter with --adapter.
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved QLoRA adapter to: {out_dir}")


if __name__ == "__main__":
    main()

