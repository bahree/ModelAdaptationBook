# Lessons and gotchas from running the book across accelerators

We validated the book's code end to end on NVIDIA (A30 Ampere, H200 Hopper, B200
Blackwell), AMD (Instinct MI300X, ROCm), and Apple Silicon (M-series, MPS). Same
code, same datasets. Here are the practical, reusable lessons that took real
debugging to find. For the per-chapter capability matrix, GPU-memory needs, and
the cross-accelerator timing comparison, see [ACCELERATORS.md](ACCELERATORS.md);
the raw per-step logs are under [`code/validation/`](code/validation/).

## 1. Pin the model to one device; do not rely on `device_map="auto"`

This is the single most important one. `device_map="auto"` is convenient on a big
GPU, but on a **memory-constrained or non-CUDA machine** it silently **offloads
layers to the meta/CPU device**, and that offload corrupts things in two different
ways we hit:

- **Adapter loading fails with a `KeyError`** (`base_model.model.model.model.…`):
  PEFT's offload bookkeeping builds a bad module name when a LoRA adapter attaches
  on top of an offloaded base. Seen in chapter 5 eval/generate on a small GPU and on a Mac.
- **Training diverges or crashes:** the same offload produced **NaN gradients on an
  Apple M2 Pro** and a **backward device-mismatch error on an M4** in the chapter 2
  quickstart.

The fix is to resolve a concrete single-device placement instead of `"auto"`:

```python
def resolve_device_map():
    if torch.cuda.is_available():
        return "auto"            # a single 24 GB+ CUDA card fits the 4B with no offload
    if torch.backends.mps.is_available():
        return {"": "mps"}       # Apple Silicon: pin everything, no offload
    return {"": "cpu"}
```

The book's `chapter05/modeling.py` uses this (`_resolve_device_map`); we applied the
same pattern to the chapter 2 quickstart. With the pin, the quickstart trains cleanly
on an M4 (loss ~3.0 → ~1.7, finite gradients), where `"auto"` had failed.

## 2. Hugging Face rate-limits some datacenter IPs (HTTP 429)

Pulling models from a cloud GPU can hit `429 Too Many Requests` from Hugging Face's
CDN, **by IP, regardless of an auth token** (we confirmed an authenticated `whoami`
still 429'd from a DigitalOcean droplet). It is the datacenter IP being throttled,
not your account. Workarounds:

- **Pre-stage the cache:** download once on an unthrottled machine and `rsync` the
  `~/.cache/huggingface/hub` directory to the box, then run with `HF_HUB_OFFLINE=1`.
- **Pull the trained model instead of training** if you only need inference (below).
- A fresh instance often gets a different, unthrottled IP.

## 3. Apple Silicon (MPS): what works, what doesn't

- **Works:** few-shot/RAG, the LoRA quickstart and chapter 5 LoRA training (with the
  device pin from lesson 1), adapter loading, and the chapter 9 monitoring tools.
- **bf16 only.** A 4B model in fp32 is ~16 GB of weights and OOMs a 16 GB Mac; load
  in bf16 (~8 GB) and let the trainer autocast in bf16.
- **QLoRA does not run on Apple Silicon** — `bitsandbytes` 4-bit kernels are
  CUDA/ROCm-only. Use the LoRA path instead.
- **Full-parameter training (chapters 6–8) OOMs** a 16 GB Mac (~18 GB needed).
- **No training GPU?** Every chapter's trained model is published to
  [`bahree/ModelAdaptationBook`](https://huggingface.co/bahree/ModelAdaptationBook);
  pull it and run inference/evaluation on the Mac without training. This is the
  recommended Apple Silicon path for the full-parameter chapters.

## 4. Blackwell (B200) works but is not yet the fast choice

- It needs the **`cu128` PyTorch wheel** (the `cu126` build lacks `sm_100`).
- Surprisingly, on this 4B workload the **B200 was the *slowest* of the three NVIDIA
  cards** — full-parameter steps ran at ~25–33 s/iter and 4-bit QLoRA dequant was
  ~190× slower per step (GPU ~7% utilized, compute-starved), because the PyTorch and
  `bitsandbytes` kernels for `sm_100` are still immature. It runs everything
  correctly; it is just not yet faster. For a 4B workload today, an H200 (or even an
  A30) is the better bet. Expect this to change as Blackwell kernels land.

## 5. AMD ROCm

The full book runs on an MI300X (ROCm), including QLoRA and full-parameter training.
`bitsandbytes` may print a warning that it could not find `rocminfo` and is defaulting
the warp size to 64 (correct for CDNA cards) — it is cosmetic; installing the ROCm
command-line tools silences it.

QLoRA's 4-bit `bitsandbytes` kernels, however, need **ROCm 6.2 or newer**. We validated
the same MI300X on both stacks: on ROCm 7.x (torch 2.10) everything runs; on an older
**ROCm 6.1** host (torch 2.6.0+rocm6.1) the LoRA path and the full-parameter chapters
(6, 7, 8) run fine, but **QLoRA fails** (24/25 overall). The PyPI `bitsandbytes` 0.49.2
wheel ships ROCm `.so` files for rocm6.2 through rocm7.x but none for rocm6.1; the 6.2
binary segfaults against the 6.1 runtime, and a source build fails because its HIP
kernels use `rocprim` block-load/store APIs absent from hipcub 6.1. It fast-fails in a
few seconds, not a hang. Use ROCm 6.2+ for the QLoRA path; LoRA and full fine-tuning are
fine on 6.1.

## 6. A few operational notes

- **The safety monitor exits with code 1 when it raises an alert** — that is by
  design (so cron/CI can detect a regression), not a failure. Treat "produced the
  report" as success; a non-zero exit with alerts is the expected signal.
- **Drift thresholds are data-specific.** Calibrate against your own train/validation
  split first; the defaults are starting points.
- **Pin the base model version** (`Qwen/Qwen3-4B-Instruct-2507`, not a floating tag)
  in training configs and the model registry, so a provider update doesn't silently
  change what you trained on.

## How we validated this

`docs/`-side maintainer tooling (`validate_all.sh`) ran the supported subset of every
chapter's code paths on each accelerator and asserted on key outputs (drift levels,
safety pass rates, adapter-load success, the HF pull-and-run path). The scrubbed
per-step logs are committed under `code/validation/<accelerator>/`.
