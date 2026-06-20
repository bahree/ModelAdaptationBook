# Book code workspace

Welcome to the code repo for **LLM Customization and Fine-Tuning: Adaptation, Distillation, and Alignment** (Manning Publications). This folder contains the runnable code, organized by chapter.

**Repository**: <https://github.com/bahree/ModelAdaptationBook>

> **Working directory:** All setup and run commands in this README must be executed from the **`code/`** directory, not the parent repo folder. If you cloned the repo and are in the root (e.g. `ModelAdaptationBook`), run `cd code` first. Running `pip install -e .` or other commands from the wrong directory will fail.

> **Which hardware do you have?** The full book runs on NVIDIA (CUDA) and AMD (ROCm); Apple Silicon (MPS) runs everything except 4-bit QLoRA and the full-parameter training chapters (6, 7, 8). For the per-chapter capability matrix, GPU memory requirements, validated environments, dependency versions, and performance across GPUs, see **[ACCELERATORS.md](../ACCELERATORS.md)** at the repo root. This README covers Python and PyTorch install; that one covers what runs where.

## Where is the Code?

The book's code is organized by chapter. Every chapter has runnable code now.

| Folder | Status | Contents |
|---|---|---|
| **`chapter01/`** | **Runnable** | Reproducibility script for the §1.6 sidebar (`run_sidebar_example.py`). Runs the same fictional-policy prompt through base Qwen3-4B, the Chapter 5 LoRA adapter, and the Chapter 6 SFT model. Base-only mode runs without Chapter 5/6 artifacts on disk. |
| **`chapter02/`** | **Runnable** | A five-step LoRA fine-tuning quick-start (`quickstart.py`) on Qwen3-4B-Instruct-2507 using a 40-example slice of the IT-support dataset: dataset prep, LoRA via TRL's `SFTTrainer`, generation, and adapter save. Runs in under 10 minutes on a 12 GB GPU and needs only the base install. Also includes `run_chapter5_adapter.py`, an optional preview that loads the chapter 5 adapter for a base-vs-adapter comparison. |
| **`chapter03/`** | **Runnable** | Data-quality experiment (`ch03_data_quality_explore.py` + helpers), six-step synthetic data pipeline (`ch03_synthetic_data_generation.py`), and `DatasetManifest` module (`ch03_datasetmanifest.py`). The experiment needs a GPU and the `chapter03` extra; the synthetic pipeline needs an Anthropic API key; the manifest module is pure stdlib. |
| **`chapter04/`** | **Runnable** | In-context learning, few-shot prompting, prompt validator, minimal RAG pipeline, retrieval-quality eval (Precision@k / Recall@k / Hit@1). CPU-friendly. |
| **`chapter05/`** | **Runnable** | LoRA and QLoRA fine-tuning of Qwen3-4B-Instruct-2507 on the IT-support dataset (Stack Exchange IT Q&A plus a Dolly mix-in). Train, evaluate, run inference, optional QLoRA. |
| **`chapter06/`** | **Runnable** | Full-parameter SFT on the same base model, with overfit monitoring, three-way base-vs-LoRA-vs-SFT eval, behavioral tests, and a safety regression suite. |
| **`chapter07/`** | **Runnable** | Black-box distillation from the chapter 6 SFT teacher into a LoRA student. Quality filtering, three-way comparison, safety robustness check, and an optional OpenRouter-backed SFT-vs-frontier comparison. |
| **`chapter08/`** | **Runnable** | DPO (Direct Preference Optimization) on the chapter 6 SFT model using TRL's `DPOTrainer`, three-way base-vs-SFT-vs-DPO comparison, safety-after-DPO check. |
| **`chapter09/`** | **Runnable** | Model registry (JSON-backed), TF-IDF drift detector, simulated rollback workflow, canary-prompt monitor, red-team safety monitor with per-category alerting. |

Shared utilities live in **`common/`** (JSONL I/O, env loading, deterministic seeding, manifest tracking, and an OpenRouter helper used by chapter 7's optional frontier comparison). Install the whole code workspace with `pip install -e .` from this directory.

For chapter-specific instructions and the listing-to-code map, see each chapter's `README.md`.

**Required Python**: **3.12+**. If you hit install issues, confirm `python3 --version` reports 3.12 or higher.

## One-time setup (Windows/macOS/Linux)

### 1. Enter the `code/` directory

Ensure you are in the `code/` folder before running any commands:

```bash
cd code   # if you're in the parent repo (ModelAdaptationBook, etc.)
```

### 2. Check Your Python Version

First, verify you have Python 3.12 or higher:

```bash
python3 --version
# Example output: Python 3.12.3
```

**If your version is below 3.12**, install a newer Python version before proceeding.

### Ubuntu/Debian Prerequisites

**If you're on Ubuntu/Debian**, install the `venv` package first:

```bash
sudo apt update
sudo apt install python3.12-venv  # or python3.10-venv, python3.11-venv depending on your version
```

### Create Virtual Environment

Create a virtual environment **inside `code/`**:

- Windows (PowerShell):
  - `py -3.12 -m venv .venv` (recommended if installed)
  - If you have multiple Pythons installed: `py -0p` to list them, then use e.g. `py -3.12 -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
- macOS/Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

Upgrade pip:

- `python -m pip install -U pip`

### Install PyTorch

**Important:** Install PyTorch separately before installing the book package. Choose the right build for your system.

**Minimum required:** PyTorch 2.0+ (recommended: PyTorch 2.10+)

**Step 1: Check if PyTorch is already installed**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**If PyTorch is installed:**
- **Version 2.0+:** You're good! Skip to "Install the package" section below
- **Version < 2.0:** Upgrade PyTorch (follow Step 2-4 below)

**If PyTorch is NOT installed** (you'll see an error):
- **Follow Steps 2-4 below** to install it

**Step 2: Check if you have a GPU and CUDA installed**

```bash
# Check CUDA version (if NVIDIA GPU installed)
nvidia-smi

# Or check nvcc version
nvcc --version
```

If `nvidia-smi` reports `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` but `lspci | grep -i nvidia` does show a GPU, the kernel module is missing. This is common on freshly provisioned Ubuntu cloud VMs and on hypervisors (Proxmox, ESXi) where the GPU is passed through. Install the matching driver from the Ubuntu archives, then reboot:

```bash
# Consumer GPUs (RTX, GTX):
sudo apt update
sudo apt install -y nvidia-driver-550

# Datacenter GPUs (A30, A100, H100, L40):
sudo apt update
sudo apt install -y nvidia-driver-550-server

sudo reboot
```

After the reboot, `nvidia-smi` should list your GPUs. The package version (550 here) controls which CUDA major version the userspace ships; PyTorch's CUDA build does not need to match exactly (see Step 3 note below).

**Step 3: Choose the right PyTorch build**

Visit the PyTorch installation page and select your configuration:
- **URL:** <https://pytorch.org/get-started/locally/>
- **Select:** Stable → Linux/Mac/Windows → Pip → Python

**Common scenarios:**

- **macOS (Apple Silicon: M1/M2/M3/M4):** no CUDA index URL, no special flags. PyTorch uses Apple's MPS (Metal) backend automatically.
  ```bash
  pip3 install torch torchvision torchaudio
  ```
  After install, verify MPS is available with:
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
  ```
  See the **macOS compatibility note** at the end of this section for which chapters run on Apple Silicon.

- **macOS (Intel Macs):** no GPU acceleration; PyTorch installs the CPU build.
  ```bash
  pip3 install torch torchvision torchaudio
  ```

- **Linux/Windows with CUDA 12.6 (most common for Ubuntu 24.04):**
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

- **Linux/Windows with CUDA 12.1 (common for Ubuntu 22.04/Windows):**
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- **Linux with an AMD GPU (ROCm):** validated on an AMD Instinct MI300X under ROCm 7.x; the LoRA quickstart, QLoRA (4-bit via `bitsandbytes`), and full-parameter SFT all trained correctly. ROCm is Linux-only and works best on datacenter (MI-series) cards; consumer RDNA support varies by GPU generation. Match the index URL to your ROCm version; the line below is what we tested.
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
  ```
  On a working ROCm install, `torch.cuda.is_available()` returns `True` (ROCm presents as CUDA in PyTorch), so the chapter preflight checks let you proceed.

- **CPU only (Linux/Windows, no GPU; testing only):**
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

**Note:** PyTorch CUDA versions don't need to exactly match your system CUDA. For example, PyTorch built for CUDA 12.1 works fine with CUDA 12.4 installed. Choose the closest available version from the PyTorch selector. The `--index-url` flag only applies to NVIDIA CUDA builds and the explicit CPU-only build; on macOS the default PyPI wheels are correct.

**Step 4: Verify PyTorch installation**

```bash
# Linux/Windows with NVIDIA GPU:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# macOS (Apple Silicon):
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output (Linux/Windows + NVIDIA):
```
PyTorch: 2.10.0+cu126  (or similar)
CUDA available: True
```

Expected output (macOS + Apple Silicon):
```
PyTorch: 2.10.0  (or similar; no +cu suffix)
MPS available: True
```

If CUDA shows `False` on a Linux/Windows machine that has an NVIDIA GPU, you may have installed the CPU version by mistake. Reinstall PyTorch with the correct CUDA build. If MPS shows `False` on an Apple Silicon Mac, your PyTorch is older than 1.12; upgrade to PyTorch 2.0+.

### macOS / AMD / what runs where

The Apple Silicon path uses Apple's MPS backend instead of CUDA; not every chapter has a Mac-friendly code path. AMD (ROCm) runs the full book on a Linux box. For the per-chapter capability matrix (NVIDIA / Apple Silicon / AMD / CPU), GPU memory requirements, validated environments, dependency versions, and performance comparisons, see **[ACCELERATORS.md](../ACCELERATORS.md)** at the repo root.

Quick rules of thumb covered there:

- **NVIDIA (CUDA) and AMD (ROCm):** full book, including chapter 5 QLoRA and the full-parameter chapters (6, 7, 8).
- **Apple Silicon (MPS):** chapters 1 (base-only), 2, 3, 4, 5 LoRA, and 9 CPU stages. **Not** chapter 5 QLoRA (`bitsandbytes` is CUDA/ROCm-only) and **not** the full-parameter chapters (they exceed a 16 GB Mac's memory).
- **CPU only:** lightweight chapters (4, 9 CPU stages, mock backends); training chapters are impractical.

If you only have a Mac, the realistic path for the training chapters is Google Colab or a cloud GPU box.

## Install the package (from the `code/` directory)

Recommended (author/dev friendly):

- `pip install -e .`

Non-editable (fine for readers):

- `pip install .`

### What does “editable” (`-e`) mean?

- **Editable** installs link your environment to your working folder, so code changes take effect immediately.
- **Non-editable** installs copy a snapshot into site-packages; if you edit code locally you must reinstall to see changes.

Optional extras:

These are **optional dependency groups** defined in `pyproject.toml`. You can install them with:

- Dev/test tooling (pytest, ruff):
  - `pip install -e ".[dev]"`
- W&B (optional, non-fatal experiment tracking):
  - `pip install -e ".[wandb]"`
- QLoRA dependencies (only needed if you want to run **chapter 5's QLoRA step**; bitsandbytes is GPU/platform dependent):
  - `pip install -e ".[qlora]"`

You can combine extras:

- `pip install -e ".[dev,wandb]"`
- `pip install -e ".[dev,wandb,qlora]"` (only if your environment supports the bitsandbytes wheel)

**Chapter 7 optional dependency: OpenRouter.** Chapter 7 includes an *optional* script that compares the chapter 6 SFT model against a frontier API (Claude, Gemini, DeepSeek, GPT) via OpenRouter. It uses `requests` (already pulled in by the core dependencies). You only need an OpenRouter API key in `code/.env` to use it; see `chapter07/README.md`.

## (Optional) Set Hugging Face Token

**Recommended but not required.** Setting a Hugging Face token enables:
- Higher rate limits for model downloads
- Faster download speeds
- Access to gated models (if needed)

**To set your token:**

1. Create a free account at <https://huggingface.co>
2. Get your token from <https://huggingface.co/settings/tokens>
3. Set it in your environment:

```bash
# Linux/macOS
export HF_TOKEN="hf_..."

# Windows PowerShell
$env:HF_TOKEN="hf_..."

# Or add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export HF_TOKEN="hf_..."' >> ~/.bashrc
```

**Note:** If you don't set a token, downloads still work but you may see warnings about unauthenticated requests. This is harmless.

## API keys (optional)

A few scripts call a hosted LLM API. **None of the core Chapter 1-5 hands-on work (LoRA/QLoRA training, evaluation, inference) needs an API key**, and the IT support dataset is already committed under `data/it_support*`, so you only need a key if you want to *rebuild the data from scratch* or run the optional generation / judging scripts.

Two keys are used, depending on what you run:

- **`OPENROUTER_API_KEY`** — the dataset build's answer-reformatting step (`scripts/reformat_it_answers.py`), the Chapter 5 Contoso capstone seed expansion (`contoso_qa_demo/expand_seed.py`), the LLM-as-judge evaluation (`scripts/eval_3lens.py`), and Chapter 7's optional frontier comparison. Create a key at <https://openrouter.ai/keys>.
- **`ANTHROPIC_API_KEY`** — only Chapter 3's synthetic-data pipeline (`chapter03/ch03_synthetic_data_generation.py`), which calls the Anthropic API directly. Create a key at <https://console.anthropic.com>.

Set whichever you need in `code/.env` (gitignored, auto-loaded by `common/env.py`), or export it in your shell:

```bash
# code/.env
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
```

If a script that needs a key can't find one, it exits with a clear message naming the variable.

## Getting Started

Once you've completed the setup above, navigate to the chapter you want to work on. Every chapter from 1 onward ships runnable code.

- **Chapter 1 (Why model adaptation?)**: `chapter01/README.md`: reproducibility script for the chapter's "What the gap actually looks like" sidebar. Inference-only.
- **Chapter 2 (How to adapt?)**: `chapter02/README.md`: five-step LoRA fine-tuning quick-start on Qwen3-4B-Instruct-2507. Needs only the base install; runs on a 12 GB GPU (or Apple Silicon MPS).
- **Chapter 3 (What data do I need?)**: `chapter03/README.md`: data-quality experiment, six-step synthetic data pipeline, and dataset manifest module. Needs `pip install -e ".[chapter03]"`; the synthetic pipeline additionally needs `ANTHROPIC_API_KEY`.
- **Chapter 4 (ICL, few-shot, RAG)**: `chapter04/README.md`: few-shot prompting, prompt validator, minimal RAG pipeline, retrieval evaluation. CPU-friendly; GPU optional.
- **Chapter 5 (LoRA & QLoRA)**: `chapter05/README.md`: data preparation, training, evaluation, inference; QLoRA branch optional.
- **Chapter 6 (Full SFT)**: `chapter06/README.md`: full-parameter fine-tuning on a technical-support subset; behavioral tests; safety regression suite.
- **Chapter 7 (Distillation)**: `chapter07/README.md`: chapter 6 SFT model as teacher, LoRA student, three-way evaluation, optional OpenRouter frontier comparison.
- **Chapter 8 (DPO and alignment)**: `chapter08/README.md`: preference data pipeline, DPO training, three-way comparison, safety after DPO.
- **Chapter 9 (Drift / registry / monitor)**: `chapter09/README.md`: model registry, drift detector, rollback demo, canary prompts, safety monitor.

**Run all chapter commands from this `code/` directory with your virtual environment activated** (e.g. `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\activate` on Windows). If you open a new terminal or reconnect via SSH, activate the venv again or you may see "No module named 'chapterNN'".

Each chapter README contains:
- Prerequisites and GPU requirements
- Step-by-step commands
- Expected output and runtime estimates
- Troubleshooting tips

**Chapter dependencies between chapters:** chapter 7 uses chapter 6's SFT checkpoint as its teacher; chapter 8 also starts from chapter 6's SFT model; chapter 9 demonstrates the operational layer using artifacts from chapters 5, 6, and 8. Complete chapter 6 before chapter 7 or chapter 8; chapter 9's stages 1, 2, and 4 are self-contained and run on any machine.

## Notes for Windows (Hugging Face cache)

- You may see a warning about **symlinks not being supported**. Caching still works; it may just use more disk space.
  - To enable symlinks, turn on Windows **Developer Mode** or run as Administrator.
  - To silence the warning: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- You may see a note about **Xet storage** and `hf_xet` not being installed. This is optional; downloads still work.
- You may see non-fatal warnings from Transformers (e.g. deprecations). These should not prevent downloads or runs.

## The training data (the IT-support dataset)

The hands-on chapters train on the book's **IT-support dataset**: real Stack Exchange IT Q&A (Super User, Ask Ubuntu, and Server Fault), filtered to genuine support topics, plus a small slice of Databricks Dolly mixed in to preserve general capability. Build it once from this `code/` directory:

```bash
# 1. Build the dataset -> data/it_support/ (train.jsonl, valid.jsonl, preferences.jsonl, manifest.json, attribution.jsonl)
python scripts/build_it_support_dataset.py

# 2. Reformat the answers into the house style -> data/it_support_fmt/train.jsonl
python scripts/reformat_it_answers.py
```

The builder depends on `beautifulsoup4` (to clean the Stack Exchange HTML) and `datasets`, both pulled in by the base install. Per-example source URLs for the Stack Exchange content are written to `data/it_support/attribution.jsonl` (dataset licenses and redistribution terms are in the [main README](../README.md#license-and-data-attribution)). The chapter 2 quickstart and chapter 5 onward load the prepared data from `data/it_support_fmt` (and `data/it_support/valid.jsonl`).

## Shared conventions across chapters

- **Module entrypoints**: we run as Python modules from `code/`, e.g. `python -m chapter05.train_lora ...`
- **Artifacts live in chapter folders**: e.g. `chapter05/runs/...`
- **Reproducibility**: scripts write small manifests (dataset subset, adapter metadata, eval reports)
- **Experiment tracking**: W&B is opt-in; scripts must run when it is disabled or not installed.

