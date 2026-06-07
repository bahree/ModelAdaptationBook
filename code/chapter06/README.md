# Chapter 6: Supervised Fine-Tuning (SFT): Maximum Expressiveness

This chapter demonstrates full-parameter SFT on **Qwen/Qwen3-4B-Instruct-2507** using the same Dolly dataset from Chapter 5. Where Chapter 5 trained a lightweight LoRA adapter (0.07% of parameters), this chapter updates every parameter in the model, enabling a direct comparison between the two approaches.

The running example throughout the chapter is an **IT technical support assistant** that answers questions about software installation, network troubleshooting, and system configuration. The dataset is built by filtering Dolly 15K for technical support categories and keywords, then formatting the results as chat-message JSONL.

## Assumptions

This README assumes you have already completed the one-time setup from [`code/README.md`](../README.md) (Python 3.10+, virtual environment, PyTorch with CUDA, `pip install -e ".[dev]"`). If not, start there first.

**GPU requirements:** Full SFT trains all ~4B parameters and needs significantly more memory than LoRA:

| GPU | VRAM | SFT feasibility |
|-----|------|-----------------|
| NVIDIA A100 (40/80 GB) | 40-80 GB | Comfortable; single GPU |
| NVIDIA A30 / RTX 4090 (24 GB) | 24 GB | Works well with gradient checkpointing |
| RTX 4070/4080 (12-16 GB) | 12-16 GB | Tight; may OOM. Use LoRA (Chapter 5) instead |

**Disk space:** The final saved model is ~7-10 GB (vs. 15-30 MB for a LoRA adapter). Intermediate epoch checkpoints, however, also store optimizer state and run ~22-24 GB each. With the default `save_total_limit=3` the run directory peaks at roughly 70-80 GB during training before the final model is written. **Plan for ~80 GB free** if you keep the default save policy; reduce `save_total_limit` if you need to cap disk usage.

## Code layout

| Location | Contents |
|----------|----------|
| `scripts/` | Runnable scripts (prepare dataset, monitor, behavioral tests, safety regression) |
| `*.py` (this folder) | Python package modules (training, eval, inference). Run as `python -m chapter06.<module>` |
| `data/` | Generated datasets (created by Step 1 below) |
| `tests/` | Unit tests for eval utilities |

Shared utilities (JSONL, env, seed) live in `code/common/`. Evaluation metrics (`token_f1`, `exact_match`, `is_refusal`) are reused from `code/chapter05/metrics.py`.

## Listing map

| Listing | Description | File |
|---------|-------------|------|
| 6.1 | Data preparation (three-stage filter pipeline) | `scripts/prepare_sft_dataset.py` |
| 6.2 | Full SFT training script | `train_sft.py` |
| 6.3 | Inference with fine-tuned model | `generate.py` |
| 6.4 | Training monitor (overfitting and gradient checks) | `scripts/monitor.py` |
| 6.5 | Evaluation (base vs. fine-tuned, per-category Token-F1) | `eval_sft.py` |
| 6.6 | Behavioral tests (safety, knowledge, format) | `scripts/behavioral_tests.py` |
| 6.7 | OpenAI / Azure OpenAI fine-tuning API (platform comparison) | N/A (API example in chapter text) |
| 6.8 | Google Vertex AI fine-tuning setup (platform comparison) | N/A (API example in chapter text) |
| 6.9 | Safety regression suite (pre-deployment sign-off) | `scripts/safety_regression.py` |

## Step-by-step instructions

Run all commands from the `code/` directory with your virtual environment activated.

```bash
cd /path/to/repo/code
source .venv/bin/activate   # Linux/macOS
# Windows:  .venv\Scripts\activate
```

### Step 1: Prepare dataset (Listing 6.1)

Filters Dolly 15K down to technical support categories, applies keyword filtering, and converts to chat-message JSONL.

```bash
python -m chapter06.scripts.prepare_sft_dataset \
    --out chapter06/data/dolly_sft
```

Output: 450 training + 50 validation examples in `chapter06/data/dolly_sft/`.

### Step 2: Train full SFT model (Listing 6.2)

Run a 2-step smoke test first to catch OOM or config errors:

```bash
python -m chapter06.train_sft \
    --train chapter06/data/dolly_sft/train.jsonl \
    --valid chapter06/data/dolly_sft/valid.jsonl \
    --out   chapter06/runs/sft_smoke \
    --max_steps 2 --report_to none
```

If the smoke test passes, clean up and run the full training:

```bash
rm -rf chapter06/runs/sft_smoke

python -m chapter06.train_sft \
    --train chapter06/data/dolly_sft/train.jsonl \
    --valid chapter06/data/dolly_sft/valid.jsonl \
    --out   chapter06/runs/sft_run1 \
    --report_to none
```

Training time: ~10 min (2x A30), ~45-60 min (A100), ~90-120 min (RTX 4090).

### Step 3: Generate a response (Listing 6.3)

```bash
python -m chapter06.generate \
    --model_dir chapter06/runs/sft_run1 \
    --prompt "How do I troubleshoot a VPN connection failure?"
```

Unlike Chapter 5 (base model + adapter), this loads the complete fine-tuned model from a single directory.

### Step 4: Monitor training (Listing 6.4)

After a successful run, the trainer state file lives inside each checkpoint subdirectory, not in the top-level output directory. Point the monitor at the latest checkpoint:

```bash
python -m chapter06.scripts.monitor chapter06/runs/sft_run1/checkpoint-87
```

(Replace `checkpoint-87` with whichever step the final epoch landed on; `ls chapter06/runs/sft_run1/` lists the available checkpoints.)

Shows training loss trajectory, validation loss trend, and gradient norm warnings. If you point at the top-level run directory you will see "No trainer_state.json found" — that is the cue to descend into a `checkpoint-N/` subdirectory.

### Step 5: Evaluate base vs. fine-tuned (Listing 6.5)

```bash
python -m chapter06.eval_sft \
    --data_dir chapter06/data/dolly_sft \
    --model_dir chapter06/runs/sft_run1 \
    --output chapter06/runs/sft_run1/eval_report.json
```

Evaluates the base model, frees GPU memory, then evaluates the fine-tuned model. Prints a per-category comparison and saves a JSON report.

### Step 6: Behavioral tests (Listing 6.6)

```bash
python -m chapter06.scripts.behavioral_tests \
    --model_dir chapter06/runs/sft_run1 \
    --also_test_base
```

Checks safety refusal, knowledge retention, and format compliance. Exit code 0 = all passed; exit code 1 = failures detected.

### Step 7: Safety regression suite (Listing 6.9)

```bash
python -m chapter06.scripts.safety_regression \
    --model_dir chapter06/runs/sft_run1 \
    --output_dir chapter06/eval/safety
```

Compares base vs. fine-tuned across four safety dimensions. Flags any category where the fine-tuned model's pass rate drops more than 10 percentage points below the base model.

## Results summary

Representative numbers from an NVIDIA A30 run with `seed=42`. Full details in `runs/sft_run1/eval_report.json`. Your run will vary within ±0.02 on F1 across hardware and library versions; the *pattern* (summarization and closed_qa gain the most, open_qa moves least, safety is preserved) is the reliable signal.

### Evaluation (Token-F1 on 50 held-out examples)

| Category | Base Model | Fine-Tuned | Delta |
|----------|-----------|------------|-------|
| Closed QA | 0.314 | 0.57-0.64 | +0.26 to +0.32 |
| Information extraction | 0.367 | 0.47-0.48 | +0.11 |
| Open QA | 0.219 | 0.26-0.26 | +0.04 |
| Summarization | 0.242 | 0.66-0.86 | +0.42 to +0.62 |
| **Overall** | **0.257** | **0.36-0.37** | **+0.10 to +0.11** |

The Base column reproduces byte-for-byte; the SFT column moves within a 0.05-0.20 band depending on the random initialization of optimizer state and the transformers/peft version. Use these ranges as the "expect this magnitude" guide, not as fixed targets.

### Safety regression

The safety regression suite reports a per-category pass rate for base and fine-tuned, and flags any category that drops more than 10 percentage points. Across the four categories (harmful-request refusal, uncertainty acknowledgment, bias check, general knowledge), expect both base and fine-tuned to land in the **70-80%** pass rate on this Dolly technical-support subset, **with no regression** — full SFT on a narrow, on-topic training set tends to preserve safety alignment far better than LoRA on a broader Dolly subset (the chapter 5 LoRA pass shows -40 to -80 pp on a different safety prompt set; see chapter 5's README).

Absolute pass rates below 100% reflect the limits of keyword-based heuristics, not actual model failures. The regression test measures *relative change* between base and fine-tuned.

## Key differences from Chapter 5

| Aspect | Chapter 5 (LoRA) | Chapter 6 (Full SFT) |
|--------|------------------|----------------------|
| Trainable parameters | ~3.4M (0.07%) | ~4B (100%) |
| Learning rate | 2e-4 | 2e-5 (10× lower) |
| GPU memory (training) | 12-16 GB | 18-28 GB (with gradient checkpointing) |
| Final-model size | 15-30 MB (adapter only) | 7-10 GB (full model) |
| Intermediate checkpoint size | same as final | 22-24 GB each (model + optimizer + scheduler) |
| PEFT config | Yes (LoRA rank 16) | None (all parameters trainable) |
| Loading for inference | Base model + adapter | Single directory (complete model) |

## W&B experiment tracking (optional)

W&B is not required. To enable it:

```bash
pip install wandb
echo 'WANDB_API_KEY=your_key_here' >> .env

python -m chapter06.train_sft \
    --train chapter06/data/dolly_sft/train.jsonl \
    --valid chapter06/data/dolly_sft/valid.jsonl \
    --out   chapter06/runs/sft_run1 \
    --report_to wandb
```

To disable: `export WANDB_DISABLED=true`

## Running tests

```bash
pytest chapter06/tests/ -v
```

## Troubleshooting

**"CUDA out of memory"**: Full SFT requires 18-28 GB VRAM with gradient checkpointing. If your GPU has less than 24 GB, use LoRA from Chapter 5. The batch size is already 1; there is nothing further to reduce.

**"No module named 'chapter06'" or "No module named 'chapter05'"**: Run from the `code/` directory with the venv activated. Chapter 6 imports metrics from `chapter05.metrics`, so the full package must be installed.

**"trainer_state.json not found"**: The monitor script needs the path to a checkpoint directory (e.g., `chapter06/runs/sft_run1/checkpoint-87`), not the top-level output directory. Use a glob: `chapter06/runs/sft_run1/checkpoint-*`.

**High gradient norms (>5.0)**: Early spikes (6-8) are normal for full SFT on 4B models. What matters is the trajectory: norms should settle to the 2-5 range within the first epoch. If norms stay above 10 and training loss is increasing, reduce the learning rate with `--lr 1e-5`.

## Preparing for Chapter 8

The SFT checkpoint from this chapter is the starting point for DPO training in Chapter 8. Before moving on, verify your output directory contains: (1) the full model checkpoint (`.safetensors` shards), (2) tokenizer files, (3) `training_args.bin`, and (4) the eval report JSON from Step 5. These four artifacts are all Chapter 8 needs.
