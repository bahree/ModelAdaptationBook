# Accelerators and environment guide

Everything about running this book's code on real hardware: which chapters run
on which accelerator, GPU memory requirements, the setups we validated, the
dependency versions, performance across GPUs, and the design insights behind it
all. The [README](README.md) covers what the book is and how to start; this file
is the hardware reference it links to. For the reusable, hard-won lessons behind
these results (pin the device instead of `device_map="auto"`, Hugging Face rate
limits, and per-accelerator gotchas), see [LESSONS.md](LESSONS.md).

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

✓ = validated to run. ✗ = does not run on that accelerator for *training* (use NVIDIA, AMD, or a cloud GPU) -- but the trained models for chapters 5 to 8 are published on Hugging Face, so you can still run their inference and evaluation on any machine (see [Running without training](#running-without-training-pull-the-model-from-hugging-face)). On Apple Silicon, training is correct but slower than on a GPU, so give it at least 16 GB of unified memory.

## Which accelerator do I need?

- **Any NVIDIA GPU with enough VRAM** runs everything; this is the reference path.
- **DGX Spark (GB10)** is validated for chapters 1 to 5 with the chapter recipe in this repo. Use a `cu130` (or newer) PyTorch wheel on GB10; older CUDA wheel indices may detect CUDA but lack `sm_121` kernels.
- **An AMD GPU on Linux (ROCm)** also runs everything, including QLoRA and the full-parameter chapters. Best on datacenter (MI-series) cards; consumer RDNA support varies by GPU generation.
- **A Mac (Apple Silicon)** is great for chapters 1 through 5's LoRA path and chapter 9, but cannot *train* QLoRA or the full-parameter chapters (6, 7, 8). For training those, use a cloud GPU.
- **No GPU, or a Mac/small card?** You can still follow chapters 5 through 8 by pulling the trained model from Hugging Face and running inference or evaluation, without training it. See [Running without training](#running-without-training-pull-the-model-from-hugging-face).

## Running without training: pull the model from Hugging Face

You do not need a training-capable GPU to follow along. Every chapter's trained artifact is published to a single repo, [`bahree/ModelAdaptationBook`](https://huggingface.co/bahree/ModelAdaptationBook), as a per-chapter subfolder. Training a full-parameter model (chapters 6 to 8) needs a CUDA 24 GB+ card, but **loading the published model and running inference or evaluation fits a single smaller GPU or Apple Silicon (MPS)**. So on a Mac you can pull, for example, the chapter 6 SFT model and run its three-way evaluation without ever training it.

| Subfolder | Chapter | Artifact | Base |
| --- | --- | --- | --- |
| `ch5-lora` | 5 | LoRA adapter | Qwen3-4B-Instruct-2507 |
| `ch6-sft` | 6 | full SFT model (standalone) | (full fine-tune) |
| `ch7-distilled` | 7 | distilled student (LoRA) | Qwen3-4B-Instruct-2507 |
| `ch8-dpo` | 8 | full DPO model (standalone) | (full fine-tune) |
| `ch8-dpo-lora` | 8 | LoRA-DPO adapter (fits one 24 GB card) | `ch6-sft` |

Load a full model with `AutoModelForCausalLM.from_pretrained("bahree/ModelAdaptationBook", subfolder="ch6-sft")`, or an adapter with `PeftModel.from_pretrained(base, "bahree/ModelAdaptationBook", subfolder="ch5-lora")`. This is why the "What runs where" table marks chapters 6 to 8 as not running on Apple Silicon: that is about *training* them. You can still *use* their results on a Mac by pulling the trained model.

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

We verified the following on Apple Silicon (an M2 Pro and an M4, both 16 GB), using PyTorch's MPS (Metal) backend: chapter 1 in base-only mode, the chapter 2 LoRA quickstart, chapter 3's data-quality experiment, chapter 4, chapter 5's LoRA training, loading a LoRA adapter for inference, chapter 9's CPU stages plus its safety monitor and canary prompts on MPS, and pulling any of the published chapter 5 through 8 models from Hugging Face for inference and evaluation. Training on MPS is slower than a GPU, so give it a Mac with at least 16 GB of unified memory.

**One gotcha that matters on a memory-constrained Mac: pin the model to MPS, do not use `device_map="auto"`.** On a 16 GB Mac, `device_map="auto"` offloads layers to the meta/CPU device, and that offload corrupts LoRA training:

- **What we saw:** NaN gradients on an M2 Pro, and a backward device-mismatch error on an M4.
- **The fix:** pin the model to a single device, `device_map={"":"mps"}` (the `_resolve_device_map` pattern the book's code uses). With the pin, the chapter 2 quickstart trains cleanly: loss ~3.0 to ~1.7 with finite gradients throughout (confirmed on an M4, macOS 26 Tahoe).

Two things still will not run on Apple Silicon regardless:

- **4-bit QLoRA:** the `bitsandbytes` 4-bit kernels are CUDA/ROCm-only.
- **Full-parameter training (chapters 6, 7, 8):** OOMs at ~18 GB on a 16 GB Mac.

For those, train on a CUDA/ROCm GPU or Colab, or pull the trained model from Hugging Face (see [Running without training](#running-without-training-pull-the-model-from-hugging-face)).

## AMD GPU notes

The PyTorch ROCm stack supports AMD GPUs on Linux, and we validated the full book on it. On an AMD Instinct MI300X (192 GB) running ROCm 7.x with the PyTorch ROCm wheel, every chapter ran end-to-end, including chapter 5 QLoRA (4-bit via `bitsandbytes`) and the full-parameter chapters (6, 7, 8). So the chapters a Mac cannot train do train on a ROCm machine. A few notes:

- **Best on datacenter (MI-series) cards.** Consumer RDNA support varies by GPU generation.
- **Benign warning:** `bitsandbytes` may report it could not find `rocminfo` and is defaulting the warp size to 64 (correct for CDNA cards like the MI300X). 4-bit training still works; installing the ROCm command-line tools silences it.
- **The fail-fast checks treat a working ROCm install as a GPU,** so they let you proceed.
- **QLoRA needs ROCm 6.2 or newer.** We validated the same MI300X on an older ROCm 6.1 host (`torch 2.6.0+rocm6.1`): the LoRA path and the full-parameter chapters run, but **chapter 5 QLoRA fails**. The PyPI `bitsandbytes` 0.49.2 wheel ships ROCm kernels for 6.2 through 7.x but none for 6.1, the 6.2 binary segfaults on the 6.1 runtime, and a from-source build hits a hipcub 6.1 API gap. It fast-fails in seconds. Use ROCm 6.2+ for QLoRA (the 7.x stack above is the tested-good configuration); everything else runs on 6.1.

## Validated environments

The code is not pinned to one accelerator. The table records the exact machines and versions we verified on, so you know what is known-good. Other versions in the same range generally work; these are simply what we tested.

| Accelerator | Machine | OS | Driver / runtime | PyTorch | Coverage |
|---|---|---|---|---|---|
| NVIDIA CUDA | DGX Spark (GB10) | Linux (aarch64) | driver 580.159.03, CUDA 13.0 | 2.13.0+cu130 | Chapters 1 to 5 validated end-to-end with chapter-oriented recipe (`smoke`, `ch1` ... `ch5`, `all`), including chapter 2 quickstart training, chapter 3 synthetic pipeline, chapter 4 HF backend, and chapter 5 LoRA smoke + eval |
| NVIDIA CUDA | A30 (24 GB) | Linux | CUDA 12.x | 2.11+cu126 | All chapters (reference platform for the book's published numbers) |
| NVIDIA CUDA | H200 (140 GB) | Ubuntu (Nebius) | driver 580.159.04 | 2.12.1+cu126 | Full book end-to-end via validate_all.sh (25/25, incl. all training) |
| NVIDIA CUDA | B200 (179 GB, Blackwell) | Ubuntu (Nebius) | driver 580.159.04 | 2.11.0+cu128 | Full book end-to-end via validate_all.sh (25/25); needs the cu128 wheel for sm_100, and 4-bit QLoRA is slow on current Blackwell kernels (see below) |
| Apple Silicon (MPS) | Apple M2 Pro, 16 GB (also M4, 16 GB) | macOS 26.3 (and 15.6) | Metal / MPS | 2.x | Ch4, Ch5 LoRA + adapter load, Ch9 (drift, registry, rollback, safety monitor, canary on MPS), and pull-and-run of the published ch5/ch6/ch8 models. Ch5 QLoRA correctly fast-fails; full-parameter Ch6/7/8 training does not fit 16 GB (pull from Hugging Face instead). |
| AMD ROCm | Instinct MI300X (192 GB) | Ubuntu 24.04 | ROCm 7.x, HIP 7.0.51831 | 2.10.0+rocm7.0 | Full book end-to-end, including Ch5 QLoRA (4-bit) and full-parameter Ch6/7/8 |
| AMD ROCm | Instinct MI300X (192 GB) | Ubuntu (RunPod) | ROCm 6.1, HIP 6.1.40091 | 2.6.0+rocm6.1 | Full book except Ch5 QLoRA via validate_all.sh (24/25); QLoRA needs ROCm 6.2+ (no `bitsandbytes` 6.1 kernel, see AMD notes) |

## DGX Spark (GB10) run details (chapters 1 to 5)

We validated the DGX Spark chapter recipe end-to-end, including each target
(`smoke`, `ch1`, `ch2`, `ch3`, `ch4`, `ch5`) and the umbrella `all` target.
Run artifacts are in `code/validation/cuda_dgx_spark/`.

Key points from the run:

- **Use cu130 wheels on GB10.** A cu126 wheel detects CUDA but warns that
  kernels are not built for `sm_121` (compute capability 12.1). The tested-good
  setup is PyTorch `2.13.0+cu130`.
- **Recipe fix landed during validation.** The chapter 2 data prep path
  originally called `reformat_it_answers.py` without required flags. The recipe
  now passes:
  `python scripts/reformat_it_answers.py --in data/it_support/train.jsonl --out data/it_support_fmt/train.jsonl`.
- **Optional OpenRouter warnings are non-fatal.** If `OPENROUTER_API_KEY` is
  unset, the reformat step logs warnings and keeps passthrough rows; the run
  still completes.
- **Known warning noise remains non-blocking.** Some scripts emit
  `torch_dtype` deprecation and generation-config warnings (`top_p`/`top_k`);
  execution and outputs are unaffected.

## Dependency versions

All accelerators run the same code on **Python 3.12** with `transformers` 4.57.6, `peft` 0.17.1, `trl` 1.5.x, `bitsandbytes` 0.49.2 (CUDA and ROCm only). The package pins `transformers>=4.47.0,<5.0`: transformers 5.x removes a symbol that `peft` imports, so a fresh install without the upper bound resolves to 5.x and breaks `import peft`. The per-accelerator PyTorch install command is in the [README Quick start](README.md#quick-start).

For **DGX Spark (GB10)** specifically, prefer a `cu130` (or newer) PyTorch wheel. The `cu126` wheel can report CUDA available but still miss `sm_121` kernels.

## Performance across GPUs

All three GPUs do a full end-to-end pass (every chapter, real 1-epoch training) with the same code and Python 3.12. Per-step wall time in seconds:

| Step | A30 (24 GB) | MI300X (192 GB) | H200 (141 GB) |
|---|---:|---:|---:|
| Ch2 quickstart | 124 | 80 | 66 |
| Ch3 data-quality experiment | 750 | 435 | 385 |
| Ch5 train LoRA | 232 | 133 | 118 |
| Ch5 eval LoRA | 1008 | n/a* | 369 |
| Ch5 train QLoRA | 296 | 178 | 162 |
| Ch5 eval QLoRA | 924 | n/a* | 359 |
| Ch6 train SFT (full-parameter) | 217 | 122 | 118 |
| Ch7 generate teacher data | 620 | 360 | 328 |
| Ch7 train student | 62 | 44 | 31 |
| Ch7 eval distillation | 895 | 368 | 390 |
| Ch8 prepare preference data | 655 | 299 | 267 |
| Ch8 train DPO | 139 | 66 | 56 |
| Ch8 eval DPO | 763 | 381 | 331 |
| **Full pass (step time)** | **~112 min** | **~53 min** | **~50 min** |

\* The MI300X Ch5 eval steps are not captured in this pass's per-step summary.

The A30 (Ampere, 2020) is roughly 2x the datacenter cards on training and ~2.5-3x on the generation-heavy eval steps. The H200 and MI300X are close to each other; a 4B model on small datasets does not stress either, so the gap between them would only open up with larger models, longer context, or bigger batches.

**DGX Spark note:** our DGX Spark validation was chapter-recipe scoped (chapters 1 to 5), not a full-book benchmark like the table above. So we do not include DGX Spark in this full-book timing matrix to avoid apples-to-oranges comparisons.

Representative recipe-scope timings observed on DGX Spark GB10:

| DGX Spark recipe step | Observed wall time |
|---|---:|
| Ch2 quickstart training | ~74 s |
| Ch5 tiny LoRA smoke training (`validate_chapter05.py`) | ~1.4 s (training runtime, after model load) |
| Typical Qwen3-4B checkpoint load (ch1-ch5 runs) | ~44 to 46 s |

Raw logs for these timings are in `code/validation/cuda_dgx_spark/Recipe_ch2.log`, `code/validation/cuda_dgx_spark/Recipe_ch5.log`, and `code/validation/cuda_dgx_spark/Recipe_all.log`.

## Cross-accelerator validation pass (validate_all.sh)

We ran the full code path on every accelerator with one harness, `docs/overview/validate_all.sh --full` (raw scrubbed logs are in `code/validation/<accel>/`). Every box produced **identical functional results**: drift baseline 0.1859 (YELLOW) and a deliberately topic-shifted 0.6855 (RED) with `kubernetes` as the top drift term, the same bias_fairness safety alert, and a clean LoRA adapter load with no offload error. So the code reproduces across Ampere, Hopper, and Blackwell (and on Apple Silicon for the inference paths). Per-step training wall time, in seconds:

DGX Spark results in this repo are tracked separately under `code/validation/cuda_dgx_spark/` because the validation scope there is chapters 1 to 5 via recipe targets rather than the full `validate_all.sh --full` matrix. Those artifacts include per-target logs (`Recipe_smoke.log`, `Recipe_ch1.log` ... `Recipe_ch5.log`, `Recipe_all.log`) and a run summary (`code/validation/cuda_dgx_spark/README.md`).

| Step | A30 (Ampere) | H200 (Hopper) | B200 (Blackwell) |
|---|---:|---:|---:|
| Ch5 LoRA train | 763 | 485 | 993 |
| Ch5 evaluate | 1230 | 484 | 667 |
| Ch5 QLoRA train (4-bit) | 959 | 581 | 1112 |
| Ch6 SFT (full-parameter) | 618 | 343 | 893 |
| Ch7 student train | 425 | 246 | 997 |
| Ch8 DPO train | 334 | 146 | 1123 |
| Ch8 LoRA-DPO train | 320 | 168 | 1160 |

The **H200 is the fastest** across the board. The **B200, despite being the newest and largest card, is the slowest on this workload today.** At the time of testing:

- PyTorch (2.11+cu128) and `bitsandbytes` do not yet have mature Blackwell (sm_100) kernels, so full-parameter steps run at roughly 25 to 33 s/iter and 4-bit QLoRA dequant is about 190x slower per step (the GPU sits around 7% utilized, compute-starved).
- It runs every chapter correctly; it is simply not yet faster, which should change as the Blackwell software stack matures.
- It needs the `cu128` PyTorch wheel (the `cu126` build lacks sm_100).

The practical takeaway for choosing hardware today: for this 4B workload an H200 (or even the older A30) is a better bet than a B200 until Blackwell kernel support lands.

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
