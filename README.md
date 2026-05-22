# Fine-Tuning and Customizing LLMs for Enterprises

Welcome to the code repository for **Fine-Tuning and Customizing LLMs for Enterprises** (working title).

This repository contains all the runnable code, data, and examples from the book, organized by chapter.

**Book Publisher:** Manning Publications (forthcoming)
**Repository:** <https://github.com/bahree/ModelAdaptationBook>

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
| **[Chapter 2: How Do I Do Model Adaptation?](code/chapter02/README.md)** | Unsloth-based fine-tuning quick-start that reproduces the Dragon LLM open-finance recipe on Qwen3-0.6B in 15-25 minutes on a single consumer GPU. End-to-end pipeline: data preparation across four HF datasets, LoRA training via TRL's `SFTTrainer`, five evaluation tests, and model export (LoRA adapter, merged 16-bit, GGUF). |
| **[Chapter 3: What Data Do I Need?](code/chapter03/README.md)** | Data-quality experiment that trains the same model on four versions of Financial PhraseBank and compares results on a held-out test set; a six-step synthetic data generation pipeline (load → prompt → generate → quality-gate → distribution-check → mix-and-save) using a frontier teacher; and a standalone `DatasetManifest` module for content hashing, lineage tracking, and retention scheduling. |
| **[Chapter 4: In-Context Learning and Few-Shot Adaptation](code/chapter04/README.md)** | Few-shot ticket classifier, prompt validator with run-to-run variability measurement, minimal RAG pipeline (50 lines), and a Precision@k / Recall@k / Hit@1 retrieval evaluator. CPU-friendly; GPU optional. |
| **[Chapter 5: Parameter-Efficient Fine-Tuning (LoRA and QLoRA)](code/chapter05/README.md)** | LoRA and QLoRA adapters trained on a 400-example Dolly subset of Qwen3-4B-Instruct-2507, evaluated against the base model with per-category Token-F1 and a safety regression suite. |
| **[Chapter 6: Supervised Fine-Tuning (SFT)](code/chapter06/README.md)** | A full-parameter SFT of Qwen3-4B-Instruct-2507 on a technical-support Dolly subset, with overfit monitoring, three-way base-vs-LoRA-vs-SFT comparison, behavioral tests, and a separate safety regression suite. |
| **[Chapter 7: Knowledge Distillation](code/chapter07/README.md)** | Black-box distillation from the chapter 6 SFT teacher into a chapter 5-style LoRA student, with quality filtering, three-way base-vs-teacher-vs-student evaluation, safety robustness check, and an optional OpenRouter-backed SFT-vs-frontier-API comparison. |
| **[Chapter 8: DPO and Advanced Alignment](code/chapter08/README.md)** | Preference-optimisation of the chapter 6 SFT model using TRL's `DPOTrainer`; three-way base-vs-SFT-vs-DPO comparison; safety regression after DPO. |
| **[Chapter 9: Managing Model Evolution, Drift, and Versioning](code/chapter09/README.md)** | A JSON-backed model registry, a TF-IDF drift detector, a simulated rollback workflow, a canary-prompt monitor, and a red-team safety monitor with per-category alerting. |

**Start here:**
1. [code/README.md](code/README.md) — set up your Python environment and install the package.
2. The chapter README for whichever chapter you are reading.

## Quick start

```bash
git clone https://github.com/bahree/ModelAdaptationBook
cd ModelAdaptationBook/code

# Set up Python 3.10+ environment and install PyTorch + the book package.
# Full instructions (including NVIDIA driver install on fresh Ubuntu/Proxmox VMs)
# are in code/README.md.

python3 -m venv .venv
source .venv/bin/activate                      # macOS/Linux
# .venv\Scripts\Activate.ps1                   # Windows PowerShell

python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"

# Smoke-test the chapter 4 code (CPU-friendly, no model download needed):
pytest chapter04/tests/ -v
```

After that, follow the chapter README for the chapter you want to run.

## GPU requirements at a glance

| Chapter | Minimum GPU | Recommended | CPU fallback |
|---|---|---|---|
| 4 (ICL/RAG) | None for mock backends; ~8 GB for the optional Qwen3-4B HF backend | 12 GB+ | Yes (mock backend / hash embedder) |
| 5 (LoRA) | 8 GB (RTX 3060/4060+) | 12 GB+ | Yes, but ~20× slower |
| 5 (QLoRA) | 6 GB | 8 GB+ | Not recommended |
| 6 (Full SFT) | 24 GB (A30 / RTX 4090) | A100 40 GB+ | No |
| 7 (Distillation) | 12 GB (LoRA student) + 24 GB to host the chapter 6 teacher | 24 GB+ | Not recommended |
| 8 (DPO) | 24 GB | A100 40 GB+ | No |
| 9 (Drift / Registry / Monitor) | None for the CPU stages (registry, drift detector, rollback demo); ~8 GB for the GPU stages (canary, safety monitor) | 12 GB+ | Yes for stages 1, 2, and 4 |

**Disk space:** budget about 50 GB free for the Hugging Face model cache plus chapter 6's run directory (full-parameter checkpoints with optimizer state are 22-24 GB each). See `code/chapter06/README.md` for the breakdown.

## About the book

This book is a practitioner's playbook for adapting large language models to specific use cases in a production setting. It covers the full customization spectrum, from prompting (chapter 4) through parameter-efficient fine-tuning (chapter 5), full supervised fine-tuning (chapter 6), distillation (chapter 7), preference optimisation (chapter 8), and the operational layer that keeps a fine-tuned model honest in production (chapter 9). Every chapter is grounded in code that reproduces on a single consumer GPU and is calibrated against real cost economics.

## Support

- Each chapter's README has a Troubleshooting section covering the most common install and runtime issues.
- `code/README.md` covers the environment-setup pitfalls (Python version, PyTorch CUDA build, NVIDIA driver install on freshly provisioned VMs).
- If you are stuck, open an issue at <https://github.com/bahree/ModelAdaptationBook/issues> with your Python version, GPU model, and the exact error message.

## License

MIT License. See [LICENSE](LICENSE).
