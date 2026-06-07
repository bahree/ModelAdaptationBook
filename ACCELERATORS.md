# Accelerators and environment guide

Everything about running this book's code on real hardware: which chapters run
on which accelerator, GPU memory requirements, the setups we validated, the
dependency versions, performance across GPUs, and the design insights behind it
all. The [README](README.md) covers what the book is and how to start; this file
is the hardware reference it links to.

The short version: the full book runs on **NVIDIA (CUDA)** and **AMD (ROCm)**.
**Apple Silicon (MPS)** runs everything except 4-bit QLoRA and the
full-parameter training chapters. A CPU-only machine runs the lightweight
chapters but is impractical for training.

## What runs where

| Chapter | NVIDIA (CUDA) | Apple Silicon (MPS) | AMD (ROCm) | Notes |
|---|:---:|:---:|:---:|---|
| 1 - sidebar reproducer | ✓ | ✓ | ✓ | Base-only mode is inference; the LoRA/SFT branches need the chapter 5/6 artifacts |
| 2 - LoRA quickstart | ✓ | ✓ | ✓ | Small adapter; the lightest training in the book |
| 3 - data-quality experiment | ✓ | ✓ | ✓ | bf16 LoRA across four data conditions |
| 4 - ICL / RAG | ✓ | ✓ | ✓ | CPU-friendly; GPU optional |
| 5 - LoRA | ✓ | ✓ | ✓ | |
| 5 - QLoRA (4-bit) | ✓ | ✗ | ✓ | `bitsandbytes` 4-bit is CUDA/ROCm-only; on a Mac use the LoRA path |
| 6 - full-parameter SFT | ✓ | ✗ | ✓ | Needs ~24 GB; a 16 GB Mac runs out of memory (~18 GB) |
| 7 - distillation | ✓ | ✗ | ✓ | Hosts the chapter 6 teacher; same memory profile as chapter 6 |
| 8 - DPO | ✓ | ✗ | ✓ | Full-model preference optimisation; same memory profile |
| 9 - drift / registry / monitor | ✓ | ✓ | ✓ | Registry, drift detector, and rollback are CPU/stdlib; canary and safety monitor are inference |

✓ = validated to run. ✗ = does not run on that accelerator (use NVIDIA, AMD, or a cloud GPU). On Apple Silicon, training is correct but slower than on a GPU, so give it at least 16 GB of unified memory.

## Which accelerator do I need?

- **Any NVIDIA GPU with enough VRAM** runs everything; this is the reference path.
- **An AMD GPU on Linux (ROCm)** also runs everything, including QLoRA and the full-parameter chapters. Best on datacenter (MI-series) cards; consumer RDNA support varies by GPU generation.
- **A Mac (Apple Silicon)** is great for chapters 1 through 5's LoRA path and chapter 9, but cannot do QLoRA or the full-parameter chapters (6, 7, 8). For those, use a cloud GPU.
- **No GPU?** The lightweight chapters (4, the chapter 9 CPU stages, mock backends) run on CPU; training chapters are impractical.

## GPU requirements at a glance

AMD (ROCm) was validated end-to-end on a datacenter card (Instinct MI300X); VRAM needs match the NVIDIA column, and consumer RDNA support varies by GPU generation. See the [AMD note](#amd-gpu-notes) below.

| Chapter | Minimum NVIDIA GPU | Recommended | AMD (ROCm) | CPU fallback | Apple Silicon (MPS) |
|---|---|---|---|---|---|
| 1 (sidebar reproducer) | None for base-only mode; 8 GB+ for the LoRA / SFT branches | 12 GB+ | Yes | Yes (base-only, slow) | Yes (base-only mode, slow on 8 GB unified memory) |
| 2 (LoRA quick-start) | 6 GB | 12 GB+ | Yes | Yes (slow) | Yes, verified on Apple M4/16 GB (quickstart trains on MPS in ~7 min) |
| 3 (data-quality experiment) | 8 GB | 12 GB+ | Yes | Synthetic-data pipeline yes; manifest module yes; full experiment slow | Yes, verified (bf16 LoRA on MPS); synthetic-data pipeline and manifest module also yes |
| 4 (ICL/RAG) | None for mock backends; ~8 GB for the optional Qwen3-4B HF backend | 12 GB+ | Yes | Yes (mock backend / hash embedder) | Yes (mock backends are CPU; HF backend uses MPS, slow on 8 GB) |
| 5 (LoRA) | 8 GB (RTX 3060/4060+) | 12 GB+ | Yes | Yes, but ~20× slower | Yes (trains on MPS) |
| 5 (QLoRA) | 6 GB | 8 GB+ | Yes (4-bit works; benign `rocminfo` warning) | Not recommended | **No** (`bitsandbytes` is CUDA-only) |
| 6 (Full SFT) | 24 GB (A30 / RTX 4090) | A100 40 GB+ | Yes | No | No |
| 7 (Distillation) | 12 GB (LoRA student) + 24 GB to host the chapter 6 teacher | 24 GB+ | Yes | Not recommended | No |
| 8 (DPO) | 24 GB | A100 40 GB+ | Yes | No | No |
| 9 (Drift / Registry / Monitor) | None for the CPU stages (registry, drift detector, rollback demo); ~8 GB for the GPU stages (canary, safety monitor) | 12 GB+ | Yes | Yes for stages 1, 2, and 4 | Yes for stages 1, 2, and 4 |

**Disk space:** budget about 50 GB free for the Hugging Face model cache plus chapter 6's run directory (full-parameter checkpoints with optimizer state are 22-24 GB each). See `code/chapter06/README.md` for the breakdown.

## Why QLoRA needs an NVIDIA or AMD GPU

QLoRA is LoRA applied on top of a 4-bit quantized base model, and that 4-bit quantization is provided by `bitsandbytes`, whose kernels exist only for CUDA and ROCm (there is no Apple Metal/MPS build). On a Mac, use chapter 5's LoRA path, which needs no `bitsandbytes` and trains on MPS. QLoRA is a technique for fitting training into limited GPU memory, not a Mac feature, so this is expected rather than a gap. `bitsandbytes` is an optional extra in the package for exactly this reason: install it only when you want the QLoRA path.

## Apple Silicon notes

We verified the following on an Apple M4 (16 GB), macOS 15.6.1, using PyTorch's MPS (Metal) backend. These run locally: chapter 1 in base-only mode, the chapter 2 LoRA quickstart, chapter 3's data-quality experiment, chapter 4, chapter 5's LoRA training, and chapter 9's CPU stages. Training on MPS is slower than on a GPU, so give it a Mac with at least 16 GB of unified memory. Two things will not run on Apple Silicon: chapter 5's QLoRA (the `bitsandbytes` 4-bit kernels are CUDA/ROCm-only, see above), and the full-parameter training in chapters 6, 7, and 8 (chapter 6 ran out of memory at roughly 18 GB on the 16 GB Mac). Run those on Google Colab, a cloud GPU, or any CUDA/ROCm machine.

## AMD GPU notes

The PyTorch ROCm stack supports AMD GPUs on Linux, and we validated the full book on it. On an AMD Instinct MI300X (192 GB) running ROCm 7.x with the PyTorch ROCm wheel, every chapter ran end-to-end, including chapter 5 QLoRA (4-bit via `bitsandbytes`) and the full-parameter training in chapters 6, 7, and 8. So the chapters that a Mac cannot run do run on a ROCm machine. It works best on datacenter (MI-series) cards; consumer RDNA support varies by GPU generation. One caveat: `bitsandbytes` may print a warning that it could not find `rocminfo` and is defaulting the warp size to 64 (correct for CDNA cards like the MI300X); 4-bit training still works, and installing the ROCm command-line tools silences it. The fail-fast checks treat a working ROCm install as a GPU, so they let you proceed.

## Validated environments

The code is not pinned to one accelerator. The table records the exact machines and versions we verified on, so you know what is known-good. Other versions in the same range generally work; these are simply what we tested.

| Accelerator | Machine | OS | Driver / runtime | PyTorch | Coverage |
|---|---|---|---|---|---|
| NVIDIA CUDA | A30 (24 GB) | Linux | CUDA 12.x | 2.11+cu126 | All chapters (reference platform for the book's published numbers) |
| NVIDIA CUDA | H200 (141 GB) | Ubuntu 22.04 | driver 590.48.01, compute 9.0 | 2.12.0+cu126 | Full book end-to-end |
| Apple Silicon (MPS) | Apple M4, 16 GB | macOS 15.6.1 | Metal / MPS | 2.12.0 | Ch1 base-only, Ch2 quickstart, Ch3 experiment, Ch4, Ch5 LoRA, Ch9 CPU stages. Not Ch5 QLoRA or full-parameter Ch6/7/8. |
| AMD ROCm | Instinct MI300X (192 GB) | Ubuntu 24.04 | ROCm 7.x, HIP 7.0.51831 | 2.10.0+rocm7.0 | Full book end-to-end, including Ch5 QLoRA (4-bit) and full-parameter Ch6/7/8 |

## Dependency versions

All accelerators run the same code on **Python 3.12** with `transformers` 4.57.6, `peft` 0.17.1, `trl` 1.5.x, `bitsandbytes` 0.49.2 (CUDA and ROCm only). The package pins `transformers>=4.47.0,<5.0`: transformers 5.x removes a symbol that `peft` imports, so a fresh install without the upper bound resolves to 5.x and breaks `import peft`. The per-accelerator PyTorch install command is in the [README Quick start](README.md#quick-start).

## Performance across GPUs

All three GPUs do a full end-to-end pass (every chapter, real 1-epoch training) with the same code and Python 3.12. Per-step wall time in seconds:

| Step | A30 (24 GB) | MI300X (192 GB) | H200 (141 GB) |
|---|---:|---:|---:|
| Ch2 quickstart | 124 | 80 | 66 |
| Ch3 data-quality experiment | 750 | 435 | 385 |
| Ch5 train LoRA | 232 | 133 | 118 |
| Ch5 eval LoRA | 1008 | re-run* | 369 |
| Ch5 train QLoRA | 296 | 178 | 162 |
| Ch5 eval QLoRA | 924 | re-run* | 359 |
| Ch6 train SFT (full-parameter) | 217 | 122 | 118 |
| Ch7 generate teacher data | 620 | 360 | 328 |
| Ch7 train student | 62 | 44 | 31 |
| Ch7 eval distillation | 895 | 368 | 390 |
| Ch8 prepare preference data | 655 | 299 | 267 |
| Ch8 train DPO | 139 | 66 | 56 |
| Ch8 eval DPO | 763 | 381 | 331 |
| **Full pass (step time)** | **~112 min** | **~53 min** | **~50 min** |

\* The MI300X run's Ch5 eval steps initially failed on a stale documented flag and were re-run separately, so they are not in that run's per-step summary.

The A30 (Ampere, 2020) is roughly 2x the datacenter cards on training and ~2.5-3x on the generation-heavy eval steps. The H200 and MI300X are close to each other; a 4B model on small datasets does not stress either, so the gap between them would only open up with larger models, longer context, or bigger batches.

## Results reproduce across accelerators

Same code, same direction of results on every GPU (small numeric differences come from sampling and data subsetting):

| Metric | A30 | MI300X | H200 |
|---|---|---|---|
| Ch3 A/B/C/D accuracy | 97 / 98 / 96 / 84% | 93 / 98 / 97 / 82% | 98 / 98 / 96 / 87% |
| Ch7 base / teacher / student F1 | 0.262 / 0.564 / 0.487 | 0.288 / 0.466 / 0.471 | 0.258 / 0.406 / 0.459 |
| Ch8 base / SFT / DPO F1 | 0.257 / 0.356 / 0.378 | 0.255 / 0.341 / 0.364 | 0.257 / 0.339 / 0.353 |

The ordering is stable: chapter 3's corrupted condition is always the worst, the chapter 7 student approaches or matches its teacher (and always beats the base), and chapter 8 ranks DPO above SFT above base.

## Insights

From running every chapter on every accelerator:

1. **Capability is a wall, not a slope.** The question that matters is "can my hardware run this at all," not "how fast." NVIDIA and AMD run the whole book; Apple Silicon's two gaps (no 4-bit `bitsandbytes` kernels, full-parameter OOM) are hard walls, not slowness.
2. **Generation dominates runtime, not training.** The longest steps are the eval and teacher-generation passes; a 1-epoch training step is much shorter. Speeding up these runs means faster decoding (batching, shorter outputs), not a bigger training GPU.
3. **QLoRA is a memory technique, not a speed one.** On a big card QLoRA was slower than plain LoRA: 4-bit quant/dequant is overhead with no memory pressure to relieve. Its value is fitting training into limited VRAM.
4. **Datacenter GPUs are interchangeable here.** H200 and MI300X finish in about the same time; the book's 4B/small-data workload does not stress either.
5. **The A30 stays the reference.** It is the realistic "what most readers have" floor and the platform behind the book's published numbers; the bigger cards confirm the code scales up cleanly.
