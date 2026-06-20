# Cross-accelerator validation logs

These are real run logs from executing the book's code end-to-end on multiple
accelerators, so you can see exactly what runs where before you rent or buy
hardware. Each subfolder holds the per-step output of the maintainer harness
`validate_all.sh` (accelerator-aware: it runs the subset of every chapter's code
paths that the hardware supports and asserts on key outputs).

| Folder | Accelerator | Validated on |
|---|---|---|
| `cuda_a30/`  | NVIDIA (CUDA, Ampere) | A30 24 GB (the book's reference platform) |
| `cuda_h200/` | NVIDIA (CUDA, Hopper) | H200 140 GB (Nebius) |
| `cuda_b200/` | NVIDIA (CUDA, Blackwell) | B200 179 GB (Nebius) |
| `mps/`       | Apple Silicon (MPS) | M2 Pro 16 GB (Scaleway) |

All three CUDA boxes pass every step (25/25) with identical functional results; see
ACCELERATORS.md for the per-step timing comparison (and why Blackwell is currently
slower than Hopper on this workload). The Mac covers the inference + pull-and-run
paths (local 4B training on MPS is numerically unstable; see ACCELERATORS.md).

What each run exercises:

- **Any accelerator (incl. CPU):** chapter 4 tests + RAG, and chapter 9's
  registry, drift detector (a healthy same-domain YELLOW baseline plus a
  deliberately topic-shifted RED run with top drift terms), and rollback demo.
- **Apple Silicon (MPS):** chapter 5 LoRA plus the adapter-load path, the
  pull-and-run path (loading the published models from
  [`bahree/ModelAdaptationBook`](https://huggingface.co/bahree/ModelAdaptationBook)
  without training), and chapter 9's safety monitor and canary prompts. QLoRA
  fails fast with a clear message (bitsandbytes is CUDA/ROCm only).
- **NVIDIA / AMD (24 GB+):** the full training path — chapter 5 LoRA and QLoRA,
  chapter 6 SFT, chapter 7 distillation, and chapter 8 DPO and LoRA-DPO.

Logs are scrubbed of host/connection details. For the capability matrix and
per-chapter GPU-memory requirements, see [ACCELERATORS.md](../../ACCELERATORS.md).
