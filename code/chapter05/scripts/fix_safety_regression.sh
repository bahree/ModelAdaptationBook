#!/bin/bash
# Fix Safety Regression: Retrain with r=8 and Compare Results
#
# This script demonstrates how to fix safety regression by retraining with a
# smaller LoRA rank (r=8 instead of r=16), which preserves more base model behavior.
#
# Total time: ~20-30 minutes on single GPU
# - Training: 15-20 minutes
# - Evaluation: 5-10 minutes

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  Safety Regression Fix Demo"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This will:"
echo "  1. Retrain LoRA adapter with r=8 (instead of r=16)"
echo "  2. Evaluate both adapters side-by-side"
echo "  3. Generate comparison report"
echo ""
echo "Expected outcome:"
echo "  - r=16: Token-F1 +13%, Safety 60% (unsafe)"
echo "  - r=8:  Token-F1 +11%, Safety 85-90% (much better!)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Navigate to code directory
cd "$(dirname "$0")/../.."

echo "Step 1/2: Training LoRA adapter with r=8..."
echo "⏱  Estimated time: 15-20 minutes"
echo ""

python -m chapter05.train_lora \
  --train chapter05/data/dolly_subset/train.jsonl \
  --valid chapter05/data/dolly_subset/valid.jsonl \
  --out chapter05/runs/dolly_lora_r8 \
  --lora_r 8 \
  --epochs 3

echo ""
echo "✓ Training complete!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Step 2/2: Evaluating both adapters (r=16 vs r=8)..."
echo "⏱  Estimated time: 5-10 minutes"
echo ""

python chapter05/scripts/listing_5_3_evaluate.py \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/dolly_lora \
  --adapter_alt chapter05/runs/dolly_lora_r8 \
  --dolly_test chapter05/data/dolly_subset/test.jsonl \
  --out chapter05/runs/eval_comparison

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: chapter05/runs/eval_comparison/"
echo ""
echo "View the report:"
echo "  cat chapter05/runs/eval_comparison/report.md"
echo ""
echo "Look for:"
echo "  - adapter (r=16): Original with safety regression"
echo "  - adapter_alt (r=8): Fixed version with better safety"
echo "  - Compare 'Safety refusal rate Δ' between the two"
echo ""
echo "Expected improvement: 60% → 85-90% safety refusal rate"
echo ""
