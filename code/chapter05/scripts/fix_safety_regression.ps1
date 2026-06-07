# Fix Safety Regression: Retrain with r=8 and Compare Results
#
# This script demonstrates how to fix safety regression by retraining with a
# smaller LoRA rank (r=8 instead of r=16), which preserves more base model behavior.
#
# Total time: ~20-30 minutes on single GPU
# - Training: 15-20 minutes
# - Evaluation: 5-10 minutes

$ErrorActionPreference = "Stop"

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Safety Regression Fix Demo" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:"
Write-Host "  1. Retrain LoRA adapter with r=8 (instead of r=16)"
Write-Host "  2. Evaluate both adapters side-by-side"
Write-Host "  3. Generate comparison report"
Write-Host ""
Write-Host "Expected outcome:"
Write-Host "  - r=16: Token-F1 +13%, Safety 60% (unsafe)" -ForegroundColor Yellow
Write-Host "  - r=8:  Token-F1 +11%, Safety 85-90% (much better!)" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Navigate to code directory
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

Write-Host "Step 1/2: Training LoRA adapter with r=8..." -ForegroundColor Cyan
Write-Host "⏱  Estimated time: 15-20 minutes" -ForegroundColor Gray
Write-Host ""

python -m chapter05.train_lora `
  --train chapter05/data/dolly_subset/train.jsonl `
  --valid chapter05/data/dolly_subset/valid.jsonl `
  --out chapter05/runs/dolly_lora_r8 `
  --lora_r 8 `
  --epochs 3

Write-Host ""
Write-Host "✓ Training complete!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Step 2/2: Evaluating both adapters (r=16 vs r=8)..." -ForegroundColor Cyan
Write-Host "⏱  Estimated time: 5-10 minutes" -ForegroundColor Gray
Write-Host ""

python chapter05/scripts/listing_5_3_evaluate.py `
  --base Qwen/Qwen3-4B-Instruct-2507 `
  --adapter chapter05/runs/dolly_lora `
  --adapter_alt chapter05/runs/dolly_lora_r8 `
  --dolly_test chapter05/data/dolly_subset/test.jsonl `
  --out chapter05/runs/eval_comparison

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✓ Complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: chapter05/runs/eval_comparison/" -ForegroundColor Yellow
Write-Host ""
Write-Host "View the report:"
Write-Host "  Get-Content chapter05/runs/eval_comparison/report.md"
Write-Host ""
Write-Host "Look for:"
Write-Host "  - adapter (r=16): Original with safety regression"
Write-Host "  - adapter_alt (r=8): Fixed version with better safety"
Write-Host "  - Compare 'Safety refusal rate Δ' between the two"
Write-Host ""
Write-Host "Expected improvement: 60% → 85-90% safety refusal rate" -ForegroundColor Green
Write-Host ""
