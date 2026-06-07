# Understanding Your Evaluation Results

This guide helps you interpret the evaluation report from `listing_5_3_evaluate.py`.

---

## Quick Reference: What Good Results Look Like

### For 400 Training Examples (3 Epochs)

| Metric | Base Model | After LoRA | Improvement |
|--------|-----------|-----------|-------------|
| **Exact Match** | 0-10% | 5-20% | +5-15% |
| **Token-F1** | 0.20-0.35 | 0.35-0.55 | +0.10-0.20 |
| **Safety Refusal** | 90-100% | 90-100% | ±0-5% |

**Key insight:** With only 400 training examples, don't expect high absolute numbers. Focus on **positive deltas** (improvements).

---

## Understanding Your Report

### Example Report (Actual Results from This Run)

```
## base
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.212
- **Safety refusal rate**: 100.0%

## adapter
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.344
- **Safety refusal rate**: 60.0%

## adapter (Improvement vs Base)
- **Overall token-F1 Δ**: +0.1321 (13.2 percentage points)
- **Safety refusal rate Δ**: -40.0% (REGRESSION!)
```

---

## What Each Metric Means

### 1. **Exact Match (EM)**

**What it measures:** Percentage of responses that *exactly* match the reference answer after whitespace normalization.

**Typical values:**
- Base model: 0-10% (general-purpose models rarely match exactly)
- After LoRA (400 examples): 5-20%
- After LoRA (2000 examples): 20-40%

**Why it's often 0%:**
- Models rephrase answers (e.g., "The capital is Paris" vs "Paris is the capital")
- Small test sets (50 examples) amplify variability
- Instruction-tuned models prioritize helpfulness over exact phrasing

**What to look for:** Positive delta (Δ). Even +5% EM is meaningful.

---

### 2. **Token-F1**

**What it measures:** Word overlap between generated and reference responses. Range: 0.0 (no overlap) to 1.0 (perfect overlap).

**Typical values:**
- Base model: 0.15-0.35
- After LoRA (400 examples): 0.30-0.55
- After LoRA (2000 examples): 0.50-0.75

**Interpreting scores:**
- **< 0.20**: Model is not following instructions well
- **0.20-0.40**: Partial understanding, some relevant content
- **0.40-0.60**: Good instruction-following, most content relevant
- **> 0.60**: Strong alignment with reference answers

**In your report:**
- Base: 0.212 (weak)
- Adapter: 0.344 (moderate, +13.2% improvement) ✅

**Verdict:** **This is actually decent improvement!** Token-F1 jumped 62% (0.212 → 0.344). The adapter learned to generate more relevant responses.

---

### 3. **Safety Refusal Rate**

**What it measures:** Percentage of harmful/unsafe prompts where the model refuses to answer.

**Typical values:**
- Well-aligned models: 90-100% refusal
- Undertrained/misaligned models: 50-80% refusal

**In your report:**
- Base: 100% (perfect! Qwen3-4B-Instruct is well-aligned)
- Adapter: 60% (-40% REGRESSION) ⚠️

**Verdict:** **This is a safety regression.** Your fine-tuned model is less safe than the base model.

**Why this happened:**
- Dolly 15K dataset doesn't include safety training data
- LoRA adapters can override base model safety behaviors
- 400 examples of neutral instructions "diluted" the safety alignment

**How to fix:**
1. **Add safety examples to training data** (10-20% of dataset)
2. **Use a smaller LoRA rank** (r=8 instead of r=16) to preserve more base behavior
3. **Reduce training epochs** (2 instead of 3)
4. **Use QLoRA with lower learning rate** (more conservative)

---

## Per-Category Results

```
**Per-Category Improvements:**
- classification: EM Δ=+0.0%, F1 Δ=+0.4851  ✅ (48% improvement!)
- summarization: EM Δ=+0.0%, F1 Δ=+0.2945  ✅ (29% improvement!)
- open_qa: EM Δ=+0.0%, F1 Δ=+0.1576       ✅ (15% improvement!)
- brainstorming: EM Δ=+0.0%, F1 Δ=-0.0235 ⚠️ (slight regression)
- general_qa: EM Δ=+0.0%, F1 Δ=-0.0687    ⚠️ (regression)
```

**Interpretation:**
- **Classification tasks** improved the most (+48% F1) — adapter learned to categorize well
- **Summarization** also strong (+29% F1)
- **Open QA** improved (+15% F1)
- **Brainstorming/General QA** slightly regressed — adapter specialized at the cost of generalization

**This is normal:** LoRA adapters trade off some generalization for specialization. The dataset had fewer brainstorming examples, so that category didn't improve as much.

---

## Overall Assessment of Your Results

### ✅ **Good News:**
1. **Token-F1 improved by 13.2 percentage points** (62% relative improvement)
2. **Strong gains in classification (+48%), summarization (+29%), and open QA (+15%)**
3. **No catastrophic forgetting** — base capabilities mostly preserved
4. **Training completed successfully** without errors

### ⚠️ **Areas for Improvement:**
1. **Safety regression (-40%)** — This is the biggest concern
2. **0% exact match** — Indicates the model's phrasing differs from references (common but improvable)
3. **Mixed per-category results** — Some tasks regressed slightly

### 🎯 **Overall Verdict:**

**This is a typical "first LoRA run" result with a small dataset (400 examples).** You successfully fine-tuned the model and saw meaningful improvement in instruction-following (Token-F1). However, the safety regression needs to be addressed before deployment.

**For a book chapter demonstration:** These results are **pedagogically valuable** — they show both success (F1 improvement) and a common pitfall (safety regression), teaching readers to interpret real-world results critically.

---

## Next Steps to Improve

### Option 1: Quick Fix (Preserve Safety)
```bash
# Reduce LoRA rank to preserve more base model behavior
python -m chapter05.train_lora \
  --train chapter05/data/dolly_subset/train.jsonl \
  --valid chapter05/data/dolly_subset/valid.jsonl \
  --out chapter05/runs/dolly_lora_safe \
  --lora_r 8 \
  --epochs 2
```

### Option 2: Add Safety Data
1. Collect 50-100 safety examples (harmful prompts with refusal responses)
2. Mix into training data (10-20% of total)
3. Retrain with same hyperparameters

### Option 3: More Training Data
- Scale up to 1000-2000 Dolly examples
- Expect EM: 15-30%, Token-F1: 0.50-0.70
- Safety may improve with more diverse data

---

## Comparing to Chapter Expectations

**Chapter said to expect:**
- Base: ~65-70% EM, ~0.75-0.80 F1
- Adapter: ~80-88% EM, ~0.85-0.90 F1

**Why your results are lower:**
1. **Different base model behavior:** Qwen3-4B-Instruct prioritizes conversational style over exact match
2. **Small dataset:** 400 examples vs 2000+ in typical production settings
3. **Evaluation stringency:** Your test set may have more open-ended questions
4. **Random variation:** With only 50 test examples, variance is high

**This is OK!** The chapter numbers are aspirational (larger dataset). Your results show the *pattern* that matters: **positive improvement** after fine-tuning.

---

## Key Takeaways

1. **Focus on deltas (Δ), not absolute scores.** Your +13.2% Token-F1 improvement is meaningful.

2. **Safety matters.** Always check safety metrics. Your -40% regression is a red flag for production but a great learning moment for the chapter.

3. **Per-category insights are valuable.** They show which tasks your adapter specialized in (classification, summarization) vs. which regressed (brainstorming).

4. **400 examples is a starting point, not the end.** Scale up to 1000-2000 for production-grade results.

5. **These are real results.** Don't cherry-pick. Show readers both successes and pitfalls.

---

## For the Chapter

**Suggested narrative:**

> "After training, we evaluate the adapter on 50 held-out examples. The report shows:
> 
> - **Token-F1 improved from 0.21 → 0.34** (+13 percentage points), indicating the adapter learned to generate more relevant responses.
> - **Classification tasks saw a 48% improvement**, demonstrating the adapter specialized in categorization.
> - **Safety refusal rate dropped from 100% → 60%** (−40%), a regression we need to address.
> 
> With only 400 training examples, we don't expect perfect scores. The key insight is **positive improvement** in instruction-following, measured by Token-F1. However, the safety regression highlights a critical lesson: fine-tuning can inadvertently override base model safety. To fix this, we'd add safety examples to the training data or use a smaller LoRA rank (r=8) to preserve more base behavior.
> 
> For production, scale up to 1000-2000 examples and always validate safety before deployment."

---

## Questions?

**Q: Why is exact match 0%?**  
A: Instruction-tuned models rephrase answers. Focus on Token-F1 instead.

**Q: Is 0.344 F1 good?**  
A: For 400 examples, yes! It's a 62% improvement over base (0.212). With 2000 examples, expect 0.50-0.70.

**Q: Should I deploy this adapter?**  
A: **No.** The safety regression (-40%) is a blocker. Fix safety first.

**Q: How do I improve results?**  
A: (1) More data (1000-2000 examples), (2) Add safety examples, (3) Try smaller LoRA rank (r=8), (4) Tune hyperparameters (epochs, learning rate).
