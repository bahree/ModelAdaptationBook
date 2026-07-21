# DGX Spark Guide (Chapters 1-5)

This guide helps a reader run Chapters 1 through 5 on NVIDIA DGX Spark (GB10) with GPU acceleration.

## 1. What matters most on DGX Spark

DGX Spark GB10 requires a GPU build of PyTorch that supports compute capability `sm_121`.

- Use PyTorch wheels from `cu130` (or newer).
- Do not use `cu126` on GB10 for this book flow.

## 2. Host preflight checks

Run these checks before creating the Python environment:

```bash
nvidia-smi
```

You should see:
- A visible NVIDIA GPU (GB10)
- A modern driver (for example, 580+)
- CUDA runtime capability compatible with CUDA 13.x

## 3. Setup from scratch

From repository root:

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install GPU PyTorch for GB10:

```bash
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```

Install book dependencies:

```bash
python -m pip install -e ".[dev,chapter03]"
```

## 4. Verify CUDA is actually working

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name', torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('cuda_matmul_ok', y.shape)
PY
```

Expected:
- `cuda True`
- `gpu_name NVIDIA GB10`
- Successful matmul output

## 5. One-time data prep used by Chapters 2 and 5

```bash
python scripts/build_it_support_dataset.py
python scripts/reformat_it_answers.py
```

## 6. Chapter-by-chapter run sequence (1 to 5)

Run from `code/` with `.venv` active.

### Chapter 1

Base-only sidebar run:

```bash
python -m chapter01.run_sidebar_example --base_only
```

Optional base+LoRA comparison after Chapter 5 smoke adapter exists:

```bash
python -m chapter01.run_sidebar_example --lora_dir chapter05/runs/validate_lora_smoke
```

### Chapter 2

Quickstart training:

```bash
python -m chapter02.quickstart
```

Optional adapter preview:

```bash
python -m chapter02.run_chapter5_adapter --use-quickstart
```

### Chapter 3

Manifest runnable path:

```bash
python -m chapter03.ch03_datasetmanifest
```

Synthetic pipeline path (requires `ANTHROPIC_API_KEY`):

```bash
python -m chapter03.ch03_synthetic_data_generation
```

### Chapter 4

HF backend (GPU):

```bash
python -m chapter04.few_shot_demo --shots 8 --backend hf --output chapter04/runs/few_shot_hf_dgx.json
```

Validator (mock backend):

```bash
python -m chapter04.prompt_validator --shots 8 --runs 3 --backend mock
```

RAG eval (hash backend):

```bash
python -m chapter04.scripts.listing_4_5_rag_eval --k 3 --backend hash
```

### Chapter 5

Chapter preflight + tiny LoRA smoke training:

```bash
python chapter05/scripts/validate_chapter05.py
```

Generation with smoke adapter:

```bash
python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 --adapter chapter05/runs/validate_lora_smoke --prompt "My laptop won't connect to office Wi-Fi after update. What should I do?"
```

Minimal base-vs-adapter smoke evaluation:

```bash
python -m chapter05.scripts.listing_5_3_evaluate \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/validate_lora_smoke \
  --dolly_test chapter05/data/smoke/valid.jsonl \
  --toy_golden chapter05/data/golden/toy_test.jsonl \
  --safety_prompts chapter05/data/golden/safety_regression_prompts.jsonl \
  --out chapter05/runs/eval_report_dgx_smoke
```

## 7. Quick regression test command

```bash
python -m pytest -q chapter04/tests chapter05/tests
```

## 8. Common DGX Spark issues and fixes

### Issue: CUDA appears available but GPU kernels are unsupported

Symptom example:
- Warning about unsupported `sm_121` / compute capability 12.1

Fix:
- Reinstall torch from `cu130` index:

```bash
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```

### Issue: `No module named chapterXX`

Fix:
- Ensure you are in `code/`
- Ensure `.venv` is activated
- Reinstall editable package:

```bash
python -m pip install -e ".[dev,chapter03]"
```

### Issue: Chapter 1 skips SFT comparison

This is expected in a Chapter 1-5 workflow because the SFT artifact is produced in Chapter 6.

## 9. Reproducible DGX Spark automation script

Use these scripts:

- `./DGX-SPARK-RECIPE.sh` (chapter-oriented: `setup`, `ch1` ... `ch5`, `all`)
- `code/validation/cuda_dgx_spark/dgx_spark_recipe.sh` (one-shot validation path)

Examples:

```bash
# Full setup + chapter 1-5 run path
./DGX-SPARK-RECIPE.sh all

# Run only chapter 4
./DGX-SPARK-RECIPE.sh ch4

# Fast smoke checks
./DGX-SPARK-RECIPE.sh smoke
```

Validation logs and run report from a real DGX Spark pass are in:

- `code/validation/cuda_dgx_spark/`
