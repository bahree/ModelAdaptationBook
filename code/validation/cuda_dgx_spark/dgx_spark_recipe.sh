#!/usr/bin/env bash
set -euo pipefail

# DGX Spark recipe for ModelAdaptationBook chapters 1-5
# Run from the repository root or from code/.

if [[ -d code ]]; then
  cd code
fi

if [[ ! -f pyproject.toml ]]; then
  echo "Expected to run in repository code/ directory containing pyproject.toml"
  exit 1
fi

echo "[1/8] Python version"
python3 --version

echo "[2/8] Create virtual environment"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

echo "[3/8] Install GB10-compatible GPU PyTorch"
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

echo "[4/8] Verify CUDA"
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

echo "[5/8] Install book package"
python -m pip install -e ".[dev,chapter03]"

echo "[6/8] Build dataset once for chapter 2/5"
python scripts/build_it_support_dataset.py
python scripts/reformat_it_answers.py

echo "[7/8] Run chapter validation path"
python -m chapter01.run_sidebar_example --base_only
python -m chapter02.quickstart
python -m chapter03.ch03_datasetmanifest
python -m chapter04.few_shot_demo --shots 8 --backend hf --output chapter04/runs/few_shot_hf_dgx.json
python -m chapter04.prompt_validator --shots 8 --runs 3 --backend mock
python chapter05/scripts/validate_chapter05.py
python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 --adapter chapter05/runs/validate_lora_smoke --prompt "My laptop won't connect to office Wi-Fi after update. What should I do?"
python -m chapter05.scripts.listing_5_3_evaluate \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/validate_lora_smoke \
  --dolly_test chapter05/data/smoke/valid.jsonl \
  --toy_golden chapter05/data/golden/toy_test.jsonl \
  --safety_prompts chapter05/data/golden/safety_regression_prompts.jsonl \
  --out chapter05/runs/eval_report_dgx_smoke

echo "[8/8] Quick test sweep"
python -m pytest -q chapter04/tests chapter05/tests

echo "DGX Spark chapter 1-5 recipe completed successfully."
