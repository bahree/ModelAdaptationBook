"""Chapter 2 quickstart: the shape of a LoRA fine-tune in five steps.

Trains a small LoRA adapter on Qwen3-4B-Instruct-2507 using a 40-example
slice of the book's IT support dataset. Runs for max_steps=20 (about 10 to 20 minutes
on a 12 GB GPU). The point of this script is to give the reader the shape
of the recipe before chapter 5 explains every knob.

This is a preview, not the chapter 5 production recipe. The defaults match
chapter 5 so that what you learn here transfers directly. Chapter 5 explains
why each default is what it is (LoRA rank, alpha, target modules, learning
rate, gradient accumulation).

Usage:
    cd /path/to/repo/code
    source .venv/bin/activate
    python -m chapter02.quickstart

Output:
    chapter02/runs/ch2_quickstart/
        adapter_config.json
        adapter_model.safetensors
        tokenizer files
        manifest.json (records seed, base model, run config)

Hardware:
    Tested on NVIDIA A30 (24 GB) in about 8 minutes.
    Should fit on any 12 GB GPU. CPU runs are not recommended (hours).
"""
from __future__ import annotations

import datetime as dt
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _has_mps() -> bool:
    """True on Apple Silicon (Metal/MPS) builds of PyTorch."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _resolve_device_map():
    """Pin placement so the 4B model is never offloaded to meta/CPU.

    device_map="auto" offloads layers on a memory-constrained or non-CUDA box;
    on Apple Silicon that offload is what corrupts LoRA training (NaN gradients,
    or a backward device-mismatch error). Pinning to a single device avoids it.
    Same pattern as chapter05/modeling.py.
    """
    if torch.cuda.is_available():
        return "auto"            # a single 24 GB+ CUDA card fits the 4B with no offload
    if _has_mps():
        return {"": "mps"}
    return {"": "cpu"}


# Defaults match chapter 5. See chapter05/__init__.py and chapter05/modeling.py.
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = "You are a helpful assistant."
SEED = 42
TRAIN_SIZE = 40
VALID_SIZE = 5
MAX_STEPS = 20
OUTPUT_DIR = Path("chapter02/runs/ch2_quickstart")

# LoRA defaults: chapter 5 section 5.5 explains why these numbers.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
]


def step1_prepare_dataset() -> tuple[HFDataset, HFDataset, List[Dict[str, Any]]]:
    """Step 1: load the book's IT-support dataset, 40 train + 5 valid + 3 demo.

    The full dataset (real Stack Exchange IT Q&A normalized to a house format,
    plus a small Dolly general-capability slice) is built by
    scripts/build_it_support_dataset.py; here we take a small slice so the
    preview run finishes in minutes.
    """
    print("Step 1: prepare dataset")
    from common.jsonl import read_jsonl
    trows = list(read_jsonl("data/it_support_fmt/train.jsonl"))
    vrows = list(read_jsonl("data/it_support/valid.jsonl"))

    def role_of(row: Dict[str, Any], role: str) -> str:
        return next(m["content"] for m in row["messages"] if m["role"] == role)

    train_ds = HFDataset.from_list([{"messages": r["messages"]} for r in trows[:TRAIN_SIZE]])
    valid_ds = HFDataset.from_list([{"messages": r["messages"]} for r in vrows[:VALID_SIZE]])
    demo_rows = [{"instruction": role_of(r, "user"), "context": "",
                  "response": role_of(r, "assistant")}
                 for r in vrows[VALID_SIZE : VALID_SIZE + 3]]
    print(f"  train={len(train_ds)} valid={len(valid_ds)} demo={len(demo_rows)}")
    return train_ds, valid_ds, demo_rows


def step2_load_model_and_lora():
    """Step 2: load Qwen3-4B and attach a LoRA adapter.

    The base weights are frozen. SFTTrainer attaches the LoRA matrices when we
    pass `peft_config` in step 3. The same call pattern is used in chapter 5.
    """
    print("Step 2: load base model and configure LoRA")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Pin the model to one device (_resolve_device_map): device_map="auto" offloads
    # layers on a 16 GB Mac, and that offload is what makes MPS LoRA training go NaN.
    # bf16 fits a 16 GB Mac (fp32 4B would not); the trainer also autocasts in bf16
    # (step 3). fp32 only as a CPU fallback.
    load_dtype = torch.bfloat16 if (torch.cuda.is_available() or _has_mps()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=_resolve_device_map(),
        dtype=load_dtype,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora_config


def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 120) -> str:
    """One-shot greedy generation, used for the before vs after demo."""
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


def step3_train(model, tokenizer, lora_config, train_ds, valid_ds) -> SFTTrainer:
    """Step 3: train for 20 steps with SFTTrainer.

    Effective batch size is 1 * 8 = 8. With 40 examples this means the run sees
    roughly five batches per epoch, so 20 steps is about four passes over the
    data. Chapter 5 explains in depth why these defaults work.
    """
    print(f"Step 3: train for {MAX_STEPS} steps")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Autocast in bf16 on CUDA (when supported) AND on Apple MPS. On MPS this is
    # essential: without it the bf16 base model's logits overflow and grad_norm
    # becomes NaN. fp16 only on older CUDA cards that lack bf16.
    cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = cuda_bf16 or _has_mps()
    use_fp16 = torch.cuda.is_available() and not cuda_bf16

    # Apple MPS has ~16 GB of unified memory and no fp32 master-weight headroom, so
    # the IT-support examples (long Stack Exchange answers) at 512 tokens push a 4B
    # model into OOM around the first eval, and a cold-start step can overflow bf16
    # into NaN. On MPS we shorten the sequence and skip the in-training eval (the
    # memory spike); CUDA keeps the full settings. A short warmup steadies the start
    # on every device.
    mps = _has_mps()
    max_len = 256 if mps else 512
    eval_strategy = "no" if mps else "steps"

    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        max_length=max_len,
        max_grad_norm=1.0,  # clip; cheap insurance against a stray exploding step
        logging_steps=5,
        eval_strategy=eval_strategy,
        eval_steps=10,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        seed=SEED,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    t0 = time.time()
    trainer.train()
    print(f"  train wall time: {time.time() - t0:.1f}s")
    return trainer


def step4_compare(model, tokenizer, demo_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Step 4: generate on three held-out prompts so the reader can eyeball the shift.

    With only 20 training steps the change is often subtle. The point is to
    show that the pipeline produces a working adapter, not to claim the model
    is now domain-tuned.
    """
    print("Step 4: compare outputs on held-out prompts")
    samples: List[Dict[str, str]] = []
    for row in demo_rows:
        ctx = row.get("context", "") or ""
        prompt = f"{ctx}\n\n{row['instruction']}" if ctx.strip() else row["instruction"]
        # "Before" = base model (adapter disabled); "after" = base + LoRA adapter.
        with model.disable_adapter():
            before = generate(model, tokenizer, prompt)
        after = generate(model, tokenizer, prompt)
        samples.append(
            {
                "instruction": row["instruction"],
                "reference": row["response"],
                "before": before,
                "after": after,
            }
        )
        print(f"\n  Q: {row['instruction'][:80]}")
        print(f"  BEFORE: {before[:160]}")
        print(f"  AFTER : {after[:200]}")
    return samples


def step5_save(trainer: SFTTrainer, tokenizer, samples: List[Dict[str, str]]) -> None:
    """Step 5: save the adapter and a manifest the reader can audit later."""
    print(f"Step 5: save adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    manifest = {
        "base_model": BASE_MODEL,
        "dataset": "it_support (Stack Exchange IT + Dolly mix-in)",
        "train_size": TRAIN_SIZE,
        "valid_size": VALID_SIZE,
        "max_steps": MAX_STEPS,
        "seed": SEED,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
        },
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "samples": samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    train_ds, valid_ds, demo_rows = step1_prepare_dataset()
    model, tokenizer, lora_config = step2_load_model_and_lora()
    trainer = step3_train(model, tokenizer, lora_config, train_ds, valid_ds)
    samples = step4_compare(trainer.model, tokenizer, demo_rows)
    step5_save(trainer, tokenizer, samples)
    print("\nDone. The adapter is in chapter02/runs/ch2_quickstart/.")
    print("Chapter 5 walks the same recipe with full training, evaluation, and analysis.")


if __name__ == "__main__":
    main()
