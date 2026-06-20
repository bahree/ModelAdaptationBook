# Chapter 3: What Data Do I Need for Model Adaptation?

![The synthetic-data pipeline: load seeds, build a prompt, generate with a teacher model, quality-gate, distribution-check, then mix and save.](images/readme_data_pipeline.png)

*The synthetic-data pipeline this chapter builds. Starting from a few real seed examples, a frontier teacher drafts new candidates, a quality gate scores them on fidelity, format, and safety, a distribution check confirms they match your real data, and the survivors are mixed in and saved with a manifest.*

The code in this folder backs two of Chapter 3's central beats:

1. **The data-quality experiment** (§3.1 "Why data quality is #1") trains the same model on four versions of the same dataset and compares the results on a held-out test set. This is the chapter's empirical argument that data quality determines model quality more than any hyperparameter. For a concrete data-quality example in *question answering* (not just classification), see the good-versus-bad answer pairs (`quality_pairs.jsonl`) in [`../contoso_qa_demo/`](../contoso_qa_demo/README.md).
2. **The six-step synthetic data pipeline** (§3.7 "Generating synthetic training data") walks load → prompt → generate → quality-gate → distribution-check → mix-and-save end to end, using a frontier teacher model.

The chapter's dataset-manifest content (DatasetManifest, lineage, retention) is implemented in a standalone module.

## Assumptions

This README assumes you have completed the one-time setup from [`code/README.md`](../README.md) (Python 3.10+, virtual environment, PyTorch with CUDA). If not, start there first.

**Chapter 3 needs additional packages not in the base install.** Two separate use cases, two separate extras:

- For the data-quality experiment (transformers + PEFT + TRL, already in the base install):

  ```bash
  pip install -e ".[chapter03]"
  ```

  This pulls `matplotlib` and `scikit-learn` (chart and metrics), `anthropic` (for the synthetic pipeline below), and `bitsandbytes` (only used if you opt into 4-bit on a CUDA GPU; the default is plain bf16, which needs no extra and runs on Apple Silicon and CPU too).

- For the synthetic data pipeline (calls a frontier teacher LLM):

  ```bash
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

  `anthropic` is in the same `chapter03` extra. The API key is required only when you actually run the teacher calls.

The DatasetManifest module (`ch03_datasetmanifest.py`) needs only the Python standard library plus the already-installed `datasets` package.

## Hardware requirements

| Use case | GPU | VRAM | Notes |
|---|---|---|---|
| Data-quality experiment | NVIDIA / Apple Silicon (MPS) / CPU | ~12-16 GB for the default bf16 4B LoRA (or ~6-8 GB with opt-in 4-bit, CUDA only) | Four short LoRA training runs; about 13 minutes total on an A30. |
| Synthetic data pipeline | None (API-driven) | n/a | Runs on any machine; cost is API calls only (~$1-3 for a full run on Claude). |
| DatasetManifest demos | None | n/a | Pure stdlib + JSON. |

## Code layout

| File | Contents |
|------|----------|
| `ch03_data_quality_explore.py` | Main script for §3.1's data-quality experiment. Defines four data-quality conditions and orchestrates train → evaluate → compare. |
| `ch03_data_quality_helpers.py` | Helpers for the experiment: load Financial PhraseBank, inject label noise, compute Cohen's Kappa, train a single condition with LoRA via transformers + PEFT + TRL, evaluate against a held-out set, print and chart results. |
| `ch03_synthetic_data_generation.py` | The six-step synthetic data pipeline end to end (load seeds → build prompt → call teacher → quality gate → distribution check → mix and save). |
| `ch03_datasetmanifest.py` | DatasetManifest dataclass (SHA-256 hash, source provenance, quality metadata), `diff_manifests`, and `check_retention_status` helpers. |

## Listing-to-code map

The chapter's listings map to these files. The chapter's data-format listings (ChatML, Alpaca, ShareGPT, Unsloth) are reference JSON snippets in the chapter text and have no corresponding Python file.

| Listing | Topic | File / function |
|---|---|---|
| 3.1 | Define the data-quality conditions | `ch03_data_quality_explore.py` — top-level experiment parameters |
| 3.2 | Understanding how data looks with each condition | `ch03_data_quality_explore.py` — preview prints |
| 3.3 | Training the same model with different quality | `ch03_data_quality_helpers.py` — `train_condition()` |
| 3.4 | Evaluating all models on the same held-out test set | `ch03_data_quality_helpers.py` — `evaluate_condition()` |
| 3.5 | Comparing the results | `ch03_data_quality_helpers.py` — `print_results_table()`, `save_accuracy_chart()` |
| 3.6 | ChatML format | *Reference JSON in chapter text; no code file.* |
| 3.7 | Alpaca format | *Reference JSON in chapter text; no code file.* |
| 3.8 | ShareGPT format and multi-turn conversation | *Reference JSON in chapter text; no code file.* |
| 3.9 | Unsloth chat template format | *Reference JSON in chapter text; no code file.* |
| 3.10 | Loading seed examples from HuggingFace | `ch03_synthetic_data_generation.py` — `load_seeds()` |
| 3.11 | Building the generation prompt with style anchoring | `ch03_synthetic_data_generation.py` — `build_generation_prompt()` |
| 3.12 | Calling the teacher model to generate candidates | `ch03_synthetic_data_generation.py` — `generate_candidates()` |
| 3.13 | Parsing candidates | `ch03_synthetic_data_generation.py` — `parse_candidates()` |
| 3.14 | Specifying the judging system prompt | `ch03_synthetic_data_generation.py` — `JUDGE_SYSTEM_PROMPT` constant |
| 3.15 | Scoring function for one example | `ch03_synthetic_data_generation.py` — `score_one_example()` |
| 3.16 | Applying the quality gate to all candidates | `ch03_synthetic_data_generation.py` — `run_quality_gate()` |
| 3.17 | Distribution alignment check | `ch03_synthetic_data_generation.py` — `check_distribution_alignment()` |
| 3.18 | Mixing synthetic with real data and saving with a manifest | `ch03_synthetic_data_generation.py` — `mix_and_save()` |
| 3.19 | DatasetManifest example | `ch03_datasetmanifest.py` — `DatasetManifest` dataclass and the `__main__` demo |

## Running the code

All commands assume you are in the `code/` directory with the venv activated.

### 1. Data-quality experiment (§3.1)

```bash
pip install -e ".[chapter03]"
python -m chapter03.ch03_data_quality_explore
```

Expected runtime: 25-35 minutes on a single GPU (four training conditions × ~6 minutes each).

Outputs are written to `./ch03_quality_experiment/`:
- `results.json` with per-condition accuracy and F1 numbers
- `accuracy_chart.png` (if matplotlib is installed) showing the four conditions side by side

### 2. Synthetic data pipeline (§3.7)

```bash
pip install -e ".[chapter03]"
export ANTHROPIC_API_KEY=sk-ant-...
python -m chapter03.ch03_synthetic_data_generation
```

Expected cost: ~$1-3 in Claude API calls for the full demo. Expected runtime: 5-10 minutes (most of it is the teacher-model calls).

Each step is self-contained; you can also import individual functions to inspect them without running the full pipeline:

```python
from chapter03.ch03_synthetic_data_generation import load_seeds, build_generation_prompt
seeds = load_seeds(n_per_category=5)
prompt = build_generation_prompt(seeds, category="positive", n_examples=10)
```

### 3. DatasetManifest demo

```bash
python -m chapter03.ch03_datasetmanifest
```

No GPU, no API key. Walks the manifest creation, diff, and retention-check examples. Useful when reading §3.10 alongside the code.

## What gets downloaded on first run

| Asset | Source | Used by |
|---|---|---|
| Financial PhraseBank | `takala/financial_phrasebank` (HF) | Data-quality experiment (sentence-level sentiment dataset). |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` (HF) | Data-quality experiment (the book's spine model; four LoRA runs in about 13 minutes on an A30). |

The synthetic data pipeline also pulls seeds from Financial PhraseBank if not already cached.

## Expected output

**Data-quality experiment**:

Four conditions, same 150 training examples each, same hyperparameters. Only the label quality differs. Representative numbers from a single-GPU bf16 run (seed 42); your numbers will vary in the last digit across hardware and library versions, but the ordering holds:

| Condition | Description | Accuracy | macro-F1 |
|---|---|---|---|
| A | AllAgree (clean: all annotators agreed) | 0.96 | 0.95 |
| B | 75Agree (good: typical production quality) | 0.98 | 0.97 |
| C | 50Agree (noisy: bare-majority agreement) | 0.96 | 0.95 |
| D | Corrupted (clean sentences, 20% of labels flipped) | 0.84 | 0.82 |

The chapter's argument: the systematically corrupted condition D drops sharply (0.84) while natural annotator disagreement (A, B, C) stays high (0.96 to 0.98), because a model tolerates fuzzy labels but cannot learn from contradictory ones. Numbers vary run to run; the relative shape is the point.

**Synthetic data pipeline**: produces 100-200 verified synthetic examples per category at the end of a single run, with a manifest JSON capturing source, hash, and quality metadata.

## Troubleshooting

**"ANTHROPIC_API_KEY not found"** → The synthetic data pipeline calls Claude; export the key first (`export ANTHROPIC_API_KEY=sk-ant-...`).

**"sklearn not installed" / Matplotlib import error** → Both are in the `chapter03` extra; install with `pip install -e ".[chapter03]"`. The script gracefully skips the chart if matplotlib is missing.

**Out of memory** → The default loads the 4B model in bf16, which wants ~12-16 GB. On a smaller CUDA GPU, set `LOAD_IN_4BIT=True` in `ch03_data_quality_helpers.py` to use 4-bit (needs `bitsandbytes`, CUDA only). On Apple Silicon, use a 16 GB+ machine.

**`Can't pickle <class 'builtins.safe_open'>` during dataset map** → Triggered by `safetensors>=0.6`. The book pins `safetensors<0.6`; reinstall with `pip install "safetensors<0.6"` if a newer version has crept in.

---

**Repository:** <https://github.com/bahree/ModelAdaptationBook>
