#!/usr/bin/env bash
set -euo pipefail

# DGX Spark chapter recipe for ModelAdaptationBook (chapters 1-5)
# Usage:
#   ./DGX-SPARK-RECIPE.sh setup
#   ./DGX-SPARK-RECIPE.sh data
#   ./DGX-SPARK-RECIPE.sh ch1|ch2|ch3|ch4|ch5
#   ./DGX-SPARK-RECIPE.sh smoke
#   ./DGX-SPARK-RECIPE.sh all

ACTION="${1:-help}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$ROOT_DIR/code"

if [[ ! -d "$CODE_DIR" ]]; then
  echo "Could not find code/ directory under: $ROOT_DIR"
  exit 1
fi

cd "$CODE_DIR"

usage() {
  cat <<'EOF'
DGX Spark chapter recipe (chapters 1-5)

Commands:
  setup   - Create venv, install cu130 torch, install package deps, verify CUDA
  data    - Build/reformat IT-support dataset used by chapter 2 and 5
  ch1     - Run chapter 1 base-only sidebar script
  ch2     - Run chapter 2 quickstart training
  ch3     - Run chapter 3 manifest demo, plus synthetic pipeline if API key exists
  ch4     - Run chapter 4 HF few-shot + validator mock + RAG eval hash
  ch5     - Run chapter 5 validate/generate/evaluate smoke path
  smoke   - Fast smoke path: ch1 + ch4 validator + ch5 validator
  all     - Full reader path: setup + data + ch1..ch5

Notes:
  - This script is intended for DGX Spark (GB10) and installs PyTorch from cu130.
  - Run from repository root: ./DGX-SPARK-RECIPE.sh <command>
EOF
}

ensure_venv() {
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
}

verify_cuda() {
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
}

do_setup() {
  echo "[setup] Checking GPU visibility"
  nvidia-smi -L

  echo "[setup] Creating/activating venv"
  ensure_venv

  echo "[setup] Installing GB10-compatible PyTorch (cu130)"
  python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

  echo "[setup] Installing book dependencies"
  python -m pip install -e ".[dev,chapter03]"

  echo "[setup] Verifying CUDA runtime"
  verify_cuda
}

do_data() {
  ensure_venv
  echo "[data] Building IT-support dataset"
  python scripts/build_it_support_dataset.py
  python scripts/reformat_it_answers.py
}

do_ch1() {
  ensure_venv
  echo "[ch1] Running base-only sidebar"
  python -m chapter01.run_sidebar_example --base_only
}

do_ch2() {
  ensure_venv
  echo "[ch2] Running quickstart training"
  python -m chapter02.quickstart
}

do_ch3() {
  ensure_venv
  echo "[ch3] Running dataset manifest demo"
  python -m chapter03.ch03_datasetmanifest

  if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[ch3] Running synthetic data pipeline"
    python -m chapter03.ch03_synthetic_data_generation
  else
    echo "[ch3] Skipping synthetic pipeline (no ANTHROPIC_API_KEY/OPENROUTER_API_KEY set)"
  fi
}

do_ch4() {
  ensure_venv
  echo "[ch4] Running few-shot (hf backend)"
  python -m chapter04.few_shot_demo --shots 8 --backend hf --output chapter04/runs/few_shot_hf_dgx.json

  echo "[ch4] Running prompt validator (mock backend)"
  python -m chapter04.prompt_validator --shots 8 --runs 3 --backend mock

  echo "[ch4] Running RAG eval (hash backend)"
  python -m chapter04.scripts.listing_4_5_rag_eval --k 3 --backend hash

  echo "[ch4] Running chapter tests"
  python -m pytest -q chapter04/tests
}

do_ch5() {
  ensure_venv
  echo "[ch5] Running chapter validator and tiny LoRA smoke train"
  python chapter05/scripts/validate_chapter05.py

  echo "[ch5] Running generate with smoke adapter"
  python -m chapter05.generate --base Qwen/Qwen3-4B-Instruct-2507 --adapter chapter05/runs/validate_lora_smoke --prompt "My laptop won't connect to office Wi-Fi after update. What should I do?"

  echo "[ch5] Running minimal base-vs-adapter eval"
  python -m chapter05.scripts.listing_5_3_evaluate \
    --base Qwen/Qwen3-4B-Instruct-2507 \
    --adapter chapter05/runs/validate_lora_smoke \
    --dolly_test chapter05/data/smoke/valid.jsonl \
    --toy_golden chapter05/data/golden/toy_test.jsonl \
    --safety_prompts chapter05/data/golden/safety_regression_prompts.jsonl \
    --out chapter05/runs/eval_report_dgx_smoke

  echo "[ch5] Running chapter tests"
  python -m pytest -q chapter05/tests
}

do_smoke() {
  ensure_venv
  echo "[smoke] Running quick smoke checks"
  python -m chapter01.run_sidebar_example --base_only
  python -m chapter04.prompt_validator --shots 8 --runs 2 --backend mock
  python chapter05/scripts/validate_chapter05.py
}

do_all() {
  do_setup
  do_data
  do_ch1
  do_ch2
  do_ch3
  do_ch4
  do_ch5
  echo "[all] Chapters 1-5 completed"
}

case "$ACTION" in
  setup) do_setup ;;
  data) do_data ;;
  ch1) do_ch1 ;;
  ch2) do_data; do_ch2 ;;
  ch3) do_ch3 ;;
  ch4) do_ch4 ;;
  ch5) do_data; do_ch5 ;;
  smoke) do_smoke ;;
  all) do_all ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $ACTION"
    usage
    exit 1
    ;;
esac
