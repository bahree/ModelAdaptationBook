#!/usr/bin/env python3
"""
ch03_data_quality_explore.py
────────────────────────────
Chapter 3, Section 3.1 — Why Data Quality Is the #1 Factor

Demonstrates how data quality alone — with volume held constant — determines
model accuracy. Four conditions, same 150 training examples each, same model,
same hyperparameters. Only the label quality differs.

Requires: ch03_data_quality_helpers.py (in the same directory)

Hardware: NVIDIA GPU with ≥ 8 GB VRAM
Time    : ~25–35 minutes (4 training runs × ~6 minutes)

Install:
    pip install datasets transformers trl peft bitsandbytes accelerate
    pip install huggingface_hub scikit-learn matplotlib
"""

import random
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

from chapter03.ch03_data_quality_helpers import (
    load_phrasebank_split,
    inject_label_noise,
    estimate_kappa,
    train_condition,
    evaluate_condition,
    print_results_table,
    save_results_json,
    save_accuracy_chart,
    LABEL_SET,
)

# ── Experiment parameters ──────────────────────────────────────────────────────
TRAIN_N     = 150    # training examples per condition — identical for all four
TEST_N      = 100    # shared held-out test set
NOISE_RATE  = 0.20   # fraction of labels flipped for Condition D
SEED        = 42
OUTPUT_DIR  = "./ch03_quality_experiment"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  Define the four data quality conditions
# ══════════════════════════════════════════════════════════════════════════════
#
# Financial PhraseBank ships with four agreement-level splits, giving us a
# natural quality ladder without any artificial manipulation.
# We add a fourth synthetic condition to test a specific hypothesis.
#
# WHY FOUR CONDITIONS?
#   A — AllAgree  : all 16 annotators agreed  -> gold standard
#   B — 75Agree   : >= 12 annotators agreed   -> good, typical production quality
#   C — 50Agree   : bare majority agreed      -> noisy, common in crowdsourced data
#   D — Corrupted : AllAgree + 20% label flips -> tests whether systematic errors
#                                                 are worse than natural disagreement

print("=" * 60)
print("Chapter 3.1 — Data Quality Impact Experiment")
print("=" * 60)
print(f"\nExperiment design:")
print(f"  Volume  : {TRAIN_N} training examples per condition (identical)")
print(f"  Test set: {TEST_N} examples from AllAgree (shared gold standard)")
print(f"  Variable: data quality only\n")


# ── Load all four PhraseBank splits ───────────────────────────────────────────
print("Loading Financial PhraseBank...")
splits = {name: load_phrasebank_split(name)
          for name in ["AllAgree", "75Agree", "50Agree"]}

rng = random.Random(SEED)

# The test set is drawn from AllAgree (highest quality) and never changes.
# Every condition is evaluated against the same 100 gold-standard examples.
all_agree_shuffled = list(splits["AllAgree"])
rng.shuffle(all_agree_shuffled)
test_set   = all_agree_shuffled[:TEST_N]
train_pool = all_agree_shuffled[TEST_N:]

# Build the four training sets
condition_A = rng.sample(train_pool,         TRAIN_N)           # clean
condition_B = rng.sample(splits["75Agree"],  TRAIN_N)           # good
condition_C = rng.sample(splits["50Agree"],  TRAIN_N)           # noisy
condition_D = inject_label_noise(                                # corrupted
    rng.sample(train_pool, TRAIN_N), noise_rate=NOISE_RATE, seed=SEED+1
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  Show what each condition looks like
# ══════════════════════════════════════════════════════════════════════════════
#
# Before touching the GPU, show the reader what the four datasets look like.
# The same sentence can appear with different labels across conditions —
# that is the data quality problem made concrete.

print("\n" + "-" * 60)
print("What each condition looks like")
print("-" * 60)

# Approximate simulated kappa noise per split (calibrated to known agreement levels)
KAPPA_NOISE_BY_COND = {"A": 0.05, "B": 0.15, "C": 0.30, "D": NOISE_RATE}

CONDITIONS = [
    ("A", "AllAgree  (clean)",       condition_A),
    ("B", "75Agree   (good)",        condition_B),
    ("C", "50Agree   (noisy)",       condition_C),
    ("D", "Corrupted (20% flipped)", condition_D),
]

for key, desc, examples in CONDITIONS:
    dist        = Counter(ex["label"] for ex in examples)
    kappa_est   = estimate_kappa(examples, noise_rate=KAPPA_NOISE_BY_COND[key], seed=99)
    n_corrupted = sum(1 for ex in examples if ex.get("corrupted", False))

    print(f"\n  Condition {key} — {desc}")
    print(f"    kappa  : ~{kappa_est:.2f}")
    print(f"    labels : pos={dist['positive']:3d}  neg={dist['negative']:3d}  "
          f"neu={dist['neutral']:3d}"
          + (f"   [{n_corrupted} labels deliberately flipped]" if n_corrupted else ""))

    # Show 2 labelled examples so the reader sees what they are training on
    for ex in examples[:2]:
        flag = "  <- FLIPPED" if ex.get("corrupted") else ""
        print(f"    [{ex['label']:8s}]  {ex['sentence'][:72]}...{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  Train one model per condition
# ══════════════════════════════════════════════════════════════════════════════
#
# Every hyperparameter is fixed in ch03_3_1_quality_experiment_helpers.py.
# The only thing that changes between calls is the training data.
# If accuracy differs, it is because of data quality — nothing else.

print("\n" + "-" * 60)
print("Training  (same model, same hyperparameters, different data quality)")
print("-" * 60)

TRAINING_SETS = {
    "A_clean":     condition_A,
    "B_good":      condition_B,
    "C_noisy":     condition_C,
    "D_corrupted": condition_D,
}

checkpoints = {}
for cond_key, examples in TRAINING_SETS.items():
    print(f"\n  [{cond_key}]")
    checkpoints[cond_key] = train_condition(cond_key, examples, OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  Evaluate all models on the same held-out test set
# ══════════════════════════════════════════════════════════════════════════════
#
# All four models are tested on the same 100 AllAgree examples.
# AllAgree as the test set means we measure against the ground truth —
# the labels everyone agreed on, not the noisy training labels.

print("\n" + "-" * 60)
print("Evaluation  (shared test set: 100 AllAgree examples)")
print("-" * 60)

CONDITION_META = {
    "A_clean":     ("AllAgree  — clean",      0.90),
    "B_good":      ("75Agree   — good",        0.75),
    "C_noisy":     ("50Agree   — noisy",       0.55),
    "D_corrupted": ("Corrupted — 20% flipped", 0.65),
}

results = {}
for cond_key, checkpoint in checkpoints.items():
    label, kappa = CONDITION_META[cond_key]
    print(f"\n  [{cond_key}] {label}")
    metrics = evaluate_condition(checkpoint, test_set)
    results[cond_key] = {
        **metrics,
        "label":   label,
        "kappa":   kappa,
        "train_n": TRAIN_N,
    }
    print(f"    accuracy: {metrics['accuracy']:.1%}   macro F1: {metrics['macro_f1']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  Compare results and extract lessons
# ══════════════════════════════════════════════════════════════════════════════

print_results_table(results)

json_path  = save_results_json(results, OUTPUT_DIR)
chart_path = save_accuracy_chart(results, OUTPUT_DIR)

print(f"\n  Results : {json_path}")
if chart_path:
    print(f"  Chart   : {chart_path}")

# ── Key lessons ────────────────────────────────────────────────────────────────
print("""
-----------------------------------------------------------------
What this experiment tells us

1. VOLUME CANNOT COMPENSATE FOR QUALITY
   All four conditions used exactly the same number of examples.
   Any accuracy difference is purely a data quality effect.

2. SYSTEMATIC ERRORS BEAT NATURAL DISAGREEMENT
   Condition D (corrupted AllAgree) likely scores below Condition C
   (natural 50% agreement) despite using cleaner sentences. A model
   tolerates fuzzy labels; it cannot learn from contradictory ones.

3. MINORITY CLASSES SUFFER MOST
   "negative" typically drops furthest. In practice those are exactly
   the high-stakes signals — fraud, downgrades, warnings — that you
   most need the model to catch.

What to do:
  -> Check kappa before collecting more data  (section 3.1)
  -> Check balance and coverage               (section 3.3)
  -> Fill gaps with verified synthetic data   (sections 3.9-3.12)
-----------------------------------------------------------------
""")
