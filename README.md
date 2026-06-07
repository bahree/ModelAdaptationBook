# LLM Customization and Fine-Tuning: Adaptation, Distillation, and Alignment

Welcome to the code repository for **LLM Customization and Fine-Tuning: Adaptation, Distillation, and Alignment** (Manning Publications).

This repository contains all the runnable code, data, and examples from the book, organized by chapter.

**Book Publisher:** Manning Publications (forthcoming)
**Repository:** <https://github.com/bahree/ModelAdaptationBook>

> **Which hardware do I need?** The full book runs on NVIDIA (CUDA) and AMD (ROCm) GPUs. Most of it also runs on Apple Silicon (MPS), except 4-bit QLoRA and the full-parameter training chapters (6, 7, and 8). See **[ACCELERATORS.md](ACCELERATORS.md)** for the per-chapter breakdown, GPU memory requirements, the setups we validated, dependency versions, and performance across GPUs.

## What is in this repo?

| Folder | Contents |
|---|---|
| `code/` | All runnable code, organized by chapter, plus shared utilities and the package `pyproject.toml`. |
| `code/common/` | Shared utilities: JSONL I/O, env loading, deterministic seeding, manifest tracking, an OpenRouter helper (for chapter 7's optional frontier comparison). |
| `code/README.md` | One-time environment setup: Python, virtual environment, PyTorch (CUDA/CPU), package install. **Start here.** |
| `code/chapterNN/README.md` | Chapter-specific instructions: prerequisites, step-by-step commands, expected outputs, troubleshooting. |

## What you can run today

Every chapter ships with runnable code. The hands-on chapters (4 through 9) reproduce the book's published numbers within run-to-run variance. Chapters 1 through 3 ship examples that anchor each chapter's claims in code the reader can verify, run, and extend.

| Chapter and topic | What you build |
|---|---|
| **[Chapter 1: Why Model Adaptation?](code/chapter01/README.md)** | A reproducibility script for the §1.6 sidebar. Runs the same prompt through base Qwen3-4B, the Chapter 5 LoRA adapter, and the Chapter 6 SFT model side by side; degrades gracefully if the later-chapter artifacts are not yet built. |
| **[Chapter 2: How Do I Do Model Adaptation?](code/chapter02/README.md)** | A five-step LoRA fine-tuning quickstart on Qwen3-4B-Instruct-2507 using a 40-example Dolly subset (TRL's `SFTTrainer` plus PEFT): dataset prep, LoRA training, generation, and adapter save. Runs in under 10 minutes on a 12 GB GPU, and on Apple Silicon via MPS. |
| **[Chapter 3: What Data Do I Need?](code/chapter03/README.md)** | Data-quality experiment that trains the same model on four versions of Financial PhraseBank and compares results on a held-out test set; a six-step synthetic data generation pipeline (load → prompt → generate → quality-gate → distribution-check → mix-and-save) using a frontier teacher; and a standalone `DatasetManifest` module for content hashing, lineage tracking, and retention scheduling. |
| **[Chapter 4: In-Context Learning and Few-Shot Adaptation](code/chapter04/README.md)** | Few-shot ticket classifier, prompt validator with run-to-run variability measurement, minimal RAG pipeline (50 lines), and a Precision@k / Recall@k / Hit@1 retrieval evaluator. CPU-friendly; GPU optional. |
| **[Chapter 5: Parameter-Efficient Fine-Tuning (LoRA and QLoRA)](code/chapter05/README.md)** | LoRA and QLoRA adapters trained on a 400-example Dolly subset of Qwen3-4B-Instruct-2507, evaluated against the base model with per-category Token-F1 and a safety regression suite. |
| **[Chapter 6: Supervised Fine-Tuning (SFT)](code/chapter06/README.md)** | A full-parameter SFT of Qwen3-4B-Instruct-2507 on a technical-support Dolly subset, with overfit monitoring, three-way base-vs-LoRA-vs-SFT comparison, behavioral tests, and a separate safety regression suite. |
| **[Chapter 7: Knowledge Distillation](code/chapter07/README.md)** | Black-box distillation from the chapter 6 SFT teacher into a chapter 5-style LoRA student, with quality filtering, three-way base-vs-teacher-vs-student evaluation, safety robustness check, and an optional OpenRouter-backed SFT-vs-frontier-API comparison. |
| **[Chapter 8: DPO and Advanced Alignment](code/chapter08/README.md)** | Preference-optimisation of the chapter 6 SFT model using TRL's `DPOTrainer`; three-way base-vs-SFT-vs-DPO comparison; safety regression after DPO. |
| **[Chapter 9: Managing Model Evolution, Drift, and Versioning](code/chapter09/README.md)** | A JSON-backed model registry, a TF-IDF drift detector, a simulated rollback workflow, a canary-prompt monitor, and a red-team safety monitor with per-category alerting. |

**Start here:**
1. [code/README.md](code/README.md): set up your Python environment and install the package.
2. The chapter README for whichever chapter you are reading.

## Quick start

**1. Clone and create a virtual environment** (Python 3.12+):

```bash
git clone https://github.com/bahree/ModelAdaptationBook
cd ModelAdaptationBook/code

python3 -m venv .venv
source .venv/bin/activate                      # macOS/Linux
# .venv\Scripts\Activate.ps1                   # Windows PowerShell

python -m pip install -U pip
```

**2. Install PyTorch for your platform** (pick the one that matches your machine):

- **NVIDIA GPU (Linux/Windows), CUDA 12.6:**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
- **macOS (Apple Silicon):** uses the MPS (Metal) backend automatically, no CUDA needed.
  ```bash
  pip install torch torchvision torchaudio
  ```
- **AMD GPU (Linux, ROCm):** validated on an MI300X (ROCm 7.x). Match the index URL to your ROCm version; the example below is what we tested.
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
  ```
- **CPU only (any platform, no GPU):**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

For other CUDA versions (12.1, 11.8) or to confirm the right command for your machine, see the official selector at <https://pytorch.org/get-started/locally/>. `code/README.md` has more detail, including NVIDIA driver install steps for fresh Ubuntu/Proxmox VMs. Not sure which accelerator runs which chapter, or how much GPU memory you need? See **[ACCELERATORS.md](ACCELERATORS.md)**.

**3. Install the book package and smoke-test:**

```bash
pip install -e ".[dev]"
pytest chapter04/tests/ -v   # CPU-friendly, no model download needed
```

After that, follow the chapter README for the chapter you want to run.

## Accelerators and environment

The full book runs on NVIDIA (CUDA) and AMD (ROCm) GPUs; most of it also runs on Apple Silicon (MPS), and the lightweight chapters run on CPU. **[ACCELERATORS.md](ACCELERATORS.md)** is the complete reference:

- **[What runs where](ACCELERATORS.md#what-runs-where)** — a chapter-by-accelerator capability matrix.
- **[GPU requirements at a glance](ACCELERATORS.md#gpu-requirements-at-a-glance)** — per-chapter VRAM needs for NVIDIA, AMD, CPU, and Apple Silicon.
- **[Validated environments](ACCELERATORS.md#validated-environments)** and **[dependency versions](ACCELERATORS.md#dependency-versions)** — the exact machines and package versions we tested.
- **[Performance across GPUs](ACCELERATORS.md#performance-across-gpus)** — A30 vs MI300X vs H200 timings, plus design insights.

Two common gotchas, both covered there: chapter 5's QLoRA needs an NVIDIA or AMD GPU ([why](ACCELERATORS.md#why-qlora-needs-an-nvidia-or-amd-gpu)), and the full-parameter chapters (6, 7, 8) need ~24 GB so they do not fit a 16 GB Mac.

## About the book

This book is a practitioner's playbook for adapting large language models to specific use cases in a production setting. It covers the full customization spectrum, from prompting (chapter 4) through parameter-efficient fine-tuning (chapter 5), full supervised fine-tuning (chapter 6), distillation (chapter 7), preference optimisation (chapter 8), and the operational layer that keeps a fine-tuned model honest in production (chapter 9). Every chapter is grounded in code that reproduces on a single consumer GPU and is calibrated against real cost economics.

## Support

- Each chapter's README has a Troubleshooting section covering the most common install and runtime issues.
- `code/README.md` covers the environment-setup pitfalls (Python version, PyTorch CUDA build, NVIDIA driver install on freshly provisioned VMs).
- If you are stuck, open an issue at <https://github.com/bahree/ModelAdaptationBook/issues> with your Python version, GPU model, and the exact error message.

## License

MIT License. See [LICENSE](LICENSE).
