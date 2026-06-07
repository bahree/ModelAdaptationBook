# Complete Safety Regression Fix (Windows PowerShell)
#
# This script runs all 4 steps to fix safety regression:
# 1. Create safety examples
# 2. Mix with task data
# 3. Retrain with mixed data
# 4. Evaluate and compare
#
# Total time: ~30 minutes (5 setup + 20 training + 5 eval)

$ErrorActionPreference = "Stop"

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Complete Safety Regression Fix" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Create 50 safety examples (harmful prompts + refusals)"
Write-Host "  2. Mix with Dolly data (80% task, 20% safety)"
Write-Host "  3. Retrain LoRA adapter"
Write-Host "  4. Evaluate and compare results"
Write-Host ""
Write-Host "Expected outcome:" -ForegroundColor Green
Write-Host "  - Safety: 60% → 85-95% (fixed!)"
Write-Host "  - Task performance: ~slight drop but acceptable"
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Navigate to code directory
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

Write-Host "Step 1/4: Creating safety examples..." -ForegroundColor Cyan
Write-Host "⏱  Time: ~1 minute" -ForegroundColor Gray
Write-Host ""

python chapter05/scripts/create_safety_examples.py `
  --out chapter05/data/safety_examples.jsonl `
  --count 50

Write-Host ""
Write-Host "✓ Safety examples created!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 2/4: Mixing safety data with task data..." -ForegroundColor Cyan
Write-Host "⏱  Time: ~1 minute" -ForegroundColor Gray
Write-Host ""

python chapter05/scripts/train_with_safety.py `
  --task_data chapter05/data/dolly_subset/train.jsonl `
  --safety_data chapter05/data/safety_examples.jsonl `
  --safety_ratio 0.2 `
  --out chapter05/data/mixed_train

Write-Host ""
Write-Host "✓ Datasets mixed!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 3/4: Training LoRA adapter with mixed data..." -ForegroundColor Cyan
Write-Host "⏱  Time: 15-20 minutes" -ForegroundColor Gray
Write-Host ""

python -m chapter05.train_lora `
  --train chapter05/data/mixed_train/train_mixed.jsonl `
  --valid chapter05/data/dolly_subset/valid.jsonl `
  --out chapter05/runs/dolly_lora_with_safety `
  --epochs 3

Write-Host ""
Write-Host "✓ Training complete!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 4/4: Evaluating and comparing results..." -ForegroundColor Cyan
Write-Host "⏱  Time: 5-10 minutes" -ForegroundColor Gray
Write-Host ""

python chapter05/scripts/listing_5_3_evaluate.py `
  --base Qwen/Qwen3-4B-Instruct-2507 `
  --adapter chapter05/runs/dolly_lora `
  --adapter_alt chapter05/runs/dolly_lora_with_safety `
  --dolly_test chapter05/data/dolly_subset/test.jsonl `
  --out chapter05/runs/eval_with_safety_fix

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✓ Complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: chapter05/runs/eval_with_safety_fix/" -ForegroundColor Yellow
Write-Host ""
Write-Host "View the report:" -ForegroundColor White
Write-Host "  Get-Content chapter05/runs/eval_with_safety_fix/report.md"
Write-Host ""
Write-Host "Compare:" -ForegroundColor White
Write-Host "  adapter (r=16, no safety): Safety = 60%" -ForegroundColor Yellow
Write-Host "  adapter_alt (with safety): Safety = ~85-95%" -ForegroundColor Green
Write-Host ""
Write-Host "Expected improvement: 60% → 85-95% safety refusal rate" -ForegroundColor Green
Write-Host ""
