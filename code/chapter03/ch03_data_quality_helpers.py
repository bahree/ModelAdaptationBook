#!/usr/bin/env python3
"""
ch03_data_quality_helpers.py
─────────────────────────────
Infrastructure for the data quality impact experiment.

This file contains all functions that are NOT about the data quality story:
  - Loading Financial PhraseBank from HuggingFace
  - Injecting synthetic label noise
  - Computing Cohen's Kappa
  - Training one model condition with LoRA via TRL + PEFT
  - Evaluating a checkpoint on a held-out set
  - Printing the results table and saving the chart

The main file (ch03_data_quality_explore.py) calls these functions.
Keeping them here means the main file stays focused on the experiment design.

Install:
    pip install datasets transformers trl peft bitsandbytes accelerate
    pip install huggingface_hub scikit-learn matplotlib
"""

import os
import json
import random
import zipfile

from huggingface_hub import hf_hub_download
from datasets import Dataset


# ── Shared constants ──────────────────────────────────────────────────────────
# These mirror the values in the main file.
# Both files import from here so they can never drift out of sync.
MODEL_NAME      = "Qwen/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH  = 512
LOAD_IN_4BIT    = False   # plain bf16 LoRA: runs on CUDA, Apple Silicon (MPS), and CPU
LORA_R          = 8
LORA_ALPHA      = 8
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
TRAIN_EPOCHS    = 1
BATCH_SIZE      = 4
GRAD_ACCUM      = 2
LEARNING_RATE   = 3e-4
MAX_TRAIN_STEPS = 150
LABEL_SET       = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = (
    "You are a financial analyst. Classify the sentiment of financial statements.\n"
    "Respond with exactly one word: positive, negative, or neutral."
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_phrasebank_split(split_name: str) -> list[dict]:
    """
    Load one agreement-level split from Financial PhraseBank (HuggingFace).

    The dataset ships with four txt files inside a zip archive, each containing
    a different minimum annotator agreement level.  Each line is:
        sentence text@label

    We rsplit on the LAST "@" in case the sentence itself contains "@".

    Args:
        split_name: "AllAgree" | "75Agree" | "66Agree" | "50Agree"

    Returns:
        List of {sentence, label} dicts
    """
    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    fname_map = {
        "AllAgree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
        "75Agree":  "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
        "66Agree":  "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
        "50Agree":  "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    }
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(fname_map[split_name]) as f:
            for line in f.read().decode("latin-1").splitlines():
                if "@" in line:
                    sentence, label = line.rsplit("@", 1)
                    rows.append({"sentence": sentence.strip(), "label": label.strip()})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(ann_a: list[str], ann_b: list[str]) -> float:
    """
    Cohen's Kappa between two annotators.

    Corrects raw agreement for the agreement expected by chance.
    κ > 0.80 = almost perfect; 0.60–0.80 = substantial; < 0.60 = poor.
    """
    assert len(ann_a) == len(ann_b)
    n      = len(ann_a)
    labels = set(ann_a) | set(ann_b)
    p_o    = sum(a == b for a, b in zip(ann_a, ann_b)) / n
    p_e    = sum((ann_a.count(l)/n)*(ann_b.count(l)/n) for l in labels)
    return round((p_o - p_e) / (1 - p_e), 3) if p_e < 1.0 else 1.0


def estimate_kappa(examples: list[dict], noise_rate: float = 0.0, seed: int = 99) -> float:
    """
    Estimate Cohen's Kappa for a dataset by simulating a second annotator.

    For the natural PhraseBank splits, uses a noise rate calibrated to the
    known agreement level.  For the synthetic corruption condition, the
    actual noise rate is used directly.

    Args:
        examples:   List of {sentence, label} dicts
        noise_rate: Fraction of labels the simulated second annotator flips
        seed:       Random seed

    Returns:
        Estimated kappa (float)
    """
    rng    = random.Random(seed)
    ann_a  = [ex["label"] for ex in examples]
    ann_b  = []
    for lbl in ann_a:
        if rng.random() < noise_rate:
            ann_b.append(rng.choice([l for l in LABEL_SET if l != lbl]))
        else:
            ann_b.append(lbl)
    return cohen_kappa(ann_a, ann_b)


# ══════════════════════════════════════════════════════════════════════════════
# NOISE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_label_noise(
    examples:   list[dict],
    noise_rate: float,
    seed:       int = 43,
) -> list[dict]:
    """
    Flip a fraction of labels to a different (wrong) label.

    Simulates a data pipeline failure: mislabelled examples caused by a buggy
    annotation tool, a copy-paste error, or an ambiguous task brief.

    The corrupted examples are tagged with 'corrupted': True so the caller
    can report exactly how many were flipped.  The tag is NOT passed to the
    model at training time.

    Args:
        examples:   Source examples (not modified in place)
        noise_rate: Fraction to flip, e.g. 0.20 = 20%
        seed:       Random seed for reproducibility

    Returns:
        New list with some labels randomly flipped
    """
    rng    = random.Random(seed)
    result = []
    for ex in examples:
        if rng.random() < noise_rate:
            wrong = [l for l in LABEL_SET if l != ex["label"]]
            result.append({**ex, "label": rng.choice(wrong), "corrupted": True})
        else:
            result.append({**ex, "corrupted": False})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CHATML FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def to_chatml(example: dict) -> dict:
    """
    Format one example as a ChatML conversation for instruction tuning.

    The assistant turn contains only the label word so evaluation is
    unambiguous (exact string match, no post-processing).
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (
                f"Classify the sentiment of this financial statement:\n"
                f"\"{example['sentence']}\""
            )},
            {"role": "assistant", "content": example["label"]},
        ]
    }


def make_hf_dataset(examples: list[dict]) -> Dataset:
    """Convert a list of examples to a HuggingFace Dataset in ChatML format."""
    return Dataset.from_list([to_chatml(ex) for ex in examples])


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _make_bnb_config():
    """Build a 4-bit BitsAndBytes quantization config (nf4 + double quant)."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def train_condition(
    condition_key: str,
    train_examples: list[dict],
    output_base:    str,
) -> str:
    """
    Fine-tune Qwen3-4B-Instruct on one experimental condition using LoRA.

    All hyperparameters are fixed (imported from the constants above) so the
    only variable between calls is the training data.  This isolation is what
    makes accuracy differences attributable to data quality.

    Args:
        condition_key:  Short identifier used to name the output directory
        train_examples: Training examples in {sentence, label} format
        output_base:    Parent directory; checkpoints go in output_base/condition_key

    Returns:
        Path to the saved checkpoint directory
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    output_dir = os.path.join(output_base, condition_key)
    os.makedirs(output_dir, exist_ok=True)

    # Fresh model load for each condition — no weight sharing between runs.
    # Default is plain bf16 (LOAD_IN_4BIT=False) so the experiment runs on CUDA,
    # Apple Silicon (MPS), and CPU. 4-bit is opt-in and CUDA-only (bitsandbytes).
    bnb_config = _make_bnb_config() if LOAD_IN_4BIT else None
    # device_map="auto" is for CUDA (multi-GPU / offload). On Apple Silicon
    # (MPS) it mis-dispatches under gradient checkpointing ("expected device
    # meta but got mps"), so place the whole model on one device instead.
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    if device_map is None:
        model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Sync both model.config and model.generation_config — transformers checks both
    model.config.pad_token_id          = tokenizer.pad_token_id
    model.config.bos_token_id          = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # For 4-bit, this is what makes gradients flow into the LoRA adapter (without
    # it, grad_norm stays 0 and nothing learns). Harmless to skip for bf16, where
    # enable_input_require_grads() below is enough for gradient checkpointing.
    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # Required so gradient checkpointing works with frozen base layers
    model.enable_input_require_grads()

    # LoRA adapter — same rank/alpha for every condition
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Apply chat template to convert role/content dicts → token strings
    def format_fn(batch):
        return {"text": [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in batch["messages"]
        ]}

    dataset = make_hf_dataset(train_examples).map(format_fn, batched=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            max_steps=MAX_TRAIN_STEPS,
            bf16=True,
            seed=42,
            dataset_text_field="text",
            report_to="none",
            logging_steps=25,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    stats = trainer.train()
    # Save the LoRA adapter and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"    training loss: {stats.training_loss:.4f}  →  saved to {output_dir}")

    # Free GPU memory before the next condition is loaded
    del model, trainer
    torch.cuda.empty_cache()

    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_condition(
    checkpoint_path: str,
    test_examples:   list[dict],
) -> dict:
    """
    Run inference on a fine-tuned checkpoint and return accuracy metrics.

    Each example is prompted with system + user turns only.  The assistant turn
    is generated by the model and compared against the gold label by exact match
    on the first word of the response.

    Args:
        checkpoint_path: Directory containing the saved LoRA adapter
        test_examples:   List of {sentence, label} dicts

    Returns:
        dict with accuracy, macro_f1, per_class_f1, and predictions list
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, f1_score

    bnb_config = _make_bnb_config() if LOAD_IN_4BIT else None

    # Load the frozen base model, then layer the saved LoRA adapter on top.
    # device_map="auto" only on CUDA; single-device on MPS/CPU (see train_condition).
    device_map = "auto" if torch.cuda.is_available() else None
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    if device_map is None:
        base_model = base_model.to("mps" if torch.backends.mps.is_available() else "cpu")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Sync both model.config and model.generation_config — transformers checks both
    model.config.pad_token_id          = tokenizer.pad_token_id
    model.config.bos_token_id          = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    # Avoid repeated warning: do not keep both max_length and max_new_tokens.
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = 5
    gen_cfg.max_length = None
    gen_cfg.do_sample = False
    gen_cfg.temperature = 1.0

    gold_labels = [ex["label"] for ex in test_examples]
    predictions = []

    for ex in test_examples:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Classify the sentiment of this financial statement:\n"
                    f"\"{ex['sentence']}\""
                )},
            ],
            tokenize=False,
            add_generation_prompt=True,  # appends the assistant turn marker
        )
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_cfg,
            )
        # Decode only the tokens the model generated (not the prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        # Take the first word; fall back to "neutral" if unrecognised
        first_word = (response.split() or ["neutral"])[0]
        predictions.append(first_word if first_word in LABEL_SET else "neutral")

    del model, base_model
    torch.cuda.empty_cache()

    # Per-class F1
    per_class = {
        lbl: round(f1_score(
            [1 if g == lbl else 0 for g in gold_labels],
            [1 if p == lbl else 0 for p in predictions],
            zero_division=0,
        ), 3)
        for lbl in LABEL_SET
    }

    return {
        "accuracy":     round(accuracy_score(gold_labels, predictions), 4),
        "macro_f1":     round(f1_score(gold_labels, predictions, average="macro", zero_division=0), 4),
        "per_class_f1": per_class,
        "predictions":  predictions,
        "gold_labels":  gold_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: dict[str, dict]) -> None:
    """
    Print a side-by-side accuracy and F1 comparison across all conditions.

    Also highlights the key insight: whether systematic corruption (Condition D)
    scores lower than natural disagreement (Condition C).
    """
    ORDER = ["A_clean", "B_good", "C_noisy", "D_corrupted"]

    print("\n" + "=" * 64)
    print(f"  {'Condition':28s}  {'κ':>5s}  {'Accuracy':>9s}  {'Macro F1':>9s}")
    print("  " + "─" * 58)
    for key in ORDER:
        r = results[key]
        print(f"  {r['label']:28s}  {r['kappa']:>5.2f}  "
              f"{r['accuracy']:>9.1%}  {r['macro_f1']:>9.3f}")

    baseline = results["A_clean"]["accuracy"]
    print("\n  Drop vs clean baseline:")
    for key in ["B_good", "C_noisy", "D_corrupted"]:
        r    = results[key]
        drop = (baseline - r["accuracy"]) * 100
        bar  = "▓" * int(drop)
        print(f"    {r['label']:28s}  −{drop:4.1f} pp  {bar}")

    # Key insight — the counterintuitive result
    d_acc = results["D_corrupted"]["accuracy"]
    c_acc = results["C_noisy"]["accuracy"]
    print("\n  Key insight:")
    if d_acc < c_acc:
        diff = (c_acc - d_acc) * 100
        print(f"    Condition D scores {diff:.1f} pp LOWER than Condition C,")
        print(f"    even though D uses cleaner base sentences.")
        print(f"    → Systematic label errors are more damaging than natural disagreement.")
    else:
        print(f"    Conditions C and D degraded similarly — both error types reduce accuracy.")

    # Per-class F1 for clean vs corrupted
    print(f"\n  Per-class F1  (clean A  vs  corrupted D):")
    print(f"  {'Class':10s}  {'A: clean':>9s}  {'D: corrupted':>12s}  {'Change':>8s}")
    print("  " + "─" * 46)
    for lbl in LABEL_SET:
        f_a = results["A_clean"]["per_class_f1"][lbl]
        f_d = results["D_corrupted"]["per_class_f1"][lbl]
        chg = (f_d - f_a) * 100
        print(f"  {lbl:10s}  {f_a:>9.3f}  {f_d:>12.3f}  {chg:>+7.1f} pp")


def save_results_json(results: dict[str, dict], output_dir: str) -> str:
    """Persist results (excluding raw prediction lists) to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    save = {
        k: {kk: vv for kk, vv in v.items()
            if kk not in ("predictions", "gold_labels")}
        for k, v in results.items()
    }
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    return path


def save_accuracy_chart(results: dict[str, dict], output_dir: str) -> str | None:
    """
    Save a Manning-style B&W bar chart of accuracy by condition.

    Uses hatch patterns instead of colour fills so the chart reproduces
    cleanly in greyscale print.  Returns the file path, or None if
    matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping chart (pip install matplotlib)")
        return None

    ORDER   = ["A_clean", "B_good", "C_noisy", "D_corrupted"]
    labels  = [results[k]["label"]        for k in ORDER]
    accs    = [results[k]["accuracy"]*100  for k in ORDER]
    kappas  = [results[k]["kappa"]         for k in ORDER]
    hatches = ["", "//", "xx", ".."]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        range(len(labels)), accs,
        color="white", edgecolor="black", linewidth=1.5,
        hatch=hatches,
    )
    for bar, acc, kappa in zip(bars, accs, kappas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%\nκ≈{kappa}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Dashed baseline from clean condition
    ax.axhline(y=accs[0], color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(3.45, accs[0]+0.4, "baseline", ha="right", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        ["A: AllAgree\n(clean)", "B: 75Agree\n(good)",
         "C: 50Agree\n(noisy)",  "D: Corrupted\n20% flips"],
        fontsize=9,
    )
    ax.set_ylabel("Accuracy on shared held-out test set (%)", fontsize=10)
    ax.set_title(
        f"Impact of data quality on fine-tuning\n"
        f"{MODEL_NAME}  ·  same volume ({results['A_clean'].get('train_n', 150)} examples) per condition",
        fontsize=11,
    )
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_by_condition.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path
