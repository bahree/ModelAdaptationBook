#!/bin/bash
# Complete Safety Regression Fix (Linux/macOS)
#
# This script runs all 4 steps to fix safety regression:
# 1. Create safety examples
# 2. Mix with task data
# 3. Retrain with mixed data
# 4. Evaluate and compare
#
# Total time: ~30 minutes (5 setup + 20 training + 5 eval)

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  Complete Safety Regression Fix"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This will:"
echo "  1. Create 50 safety examples (harmful prompts + refusals)"
echo "  2. Mix with Dolly data (80% task, 20% safety)"
echo "  3. Retrain LoRA adapter"
echo "  4. Evaluate and compare results"
echo ""
echo "Expected outcome:"
echo "  - Safety: 60% → 85-95% (fixed!)"
echo "  - Task performance: ~slight drop but acceptable"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Navigate to code directory
cd "$(dirname "$0")/../.."

echo "Step 1/4: Creating safety examples..."
echo "⏱  Time: ~1 minute"
echo ""

python chapter05/scripts/create_safety_examples.py \
  --out chapter05/data/safety_examples.jsonl \
  --count 50

echo ""
echo "✓ Safety examples created!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "Step 2/4: Mixing safety data with task data..."
echo "⏱  Time: ~1 minute"
echo ""

python chapter05/scripts/train_with_safety.py \
  --task_data chapter05/data/dolly_subset/train.jsonl \
  --safety_data chapter05/data/safety_examples.jsonl \
  --safety_ratio 0.2 \
  --out chapter05/data/mixed_train

echo ""
echo "✓ Datasets mixed!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "Step 3/4: Training LoRA adapter with mixed data..."
echo "⏱  Time: 15-20 minutes"
echo ""

python -m chapter05.train_lora \
  --train chapter05/data/mixed_train/train_mixed.jsonl \
  --valid chapter05/data/dolly_subset/valid.jsonl \
  --out chapter05/runs/dolly_lora_with_safety \
  --epochs 3

echo ""
echo "✓ Training complete!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "Step 4/4: Evaluating and comparing results..."
echo "⏱  Time: 5-10 minutes"
echo ""

python chapter05/scripts/listing_5_3_evaluate.py \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/dolly_lora \
  --adapter_alt chapter05/runs/dolly_lora_with_safety \
  --dolly_test chapter05/data/dolly_subset/test.jsonl \
  --out chapter05/runs/eval_with_safety_fix

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: chapter05/runs/eval_with_safety_fix/"
echo ""
echo "View the report:"
echo "  cat chapter05/runs/eval_with_safety_fix/report.md"
echo ""
echo "Compare:"
echo "  adapter (r=16, no safety): Safety = 60%"
echo "  adapter_alt (with safety): Safety = ~85-95%"
echo ""
echo "Expected improvement: 60% → 85-95% safety refusal rate"
echo ""
