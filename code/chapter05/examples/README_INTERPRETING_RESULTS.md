# Understanding Your Evaluation Results

This guide helps you interpret the evaluation report from `listing_5_3_evaluate.py` on the IT support dataset.

---

## Quick Reference: What These Results Look Like

### For 450 Training Examples (3 Epochs), IT Support Dataset

| Metric | Base Model | After LoRA | Change |
|--------|-----------|-----------|--------|
| **Exact Match** | 0% | 0% | flat (models paraphrase) |
| **Token-F1** | ~0.15-0.16 | ~0.15-0.16 | roughly flat |
| **Safety Refusal** | 100% | 60% | regression (helpful-only data) |

**Key insight:** On long generative IT answers, token-F1 barely moves and exact match stays at zero. The metric that actually changes is the **safety refusal rate**, and it changes for the worse.

---

## Understanding Your Report

### Example Report (Actual Results from This Run)

```
## base
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.158
- **Safety refusal rate**: 100.0%

## adapter
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.156
- **Safety refusal rate**: 60.0%

## adapter (Improvement vs Base)
- **Overall token-F1 Δ**: -0.0014 (essentially flat)
- **Safety refusal rate Δ**: -40.0% (REGRESSION)
```

---

## What Each Metric Means

### 1. Exact Match (EM)

**What it measures:** Percentage of responses that *exactly* match the reference answer after whitespace normalization.

**What you see here:** 0% for base and adapter alike.

**Why it's 0%:**
- IT answers are long and open-ended; the model paraphrases instead of copying.
- "Restart the VPN client and update it" and "Update your VPN client, then restart it" are both correct but share almost no exact phrasing.
- Instruction-tuned models prioritize a helpful answer over matching a reference string.

**What to look for:** Don't read into exact match for this task. It is the wrong tool for long generative answers.

---

### 2. Token-F1

**What it measures:** Word overlap between generated and reference responses. Range: 0.0 (no overlap) to 1.0 (perfect overlap).

**What you see here:**
- Base: 0.158
- Adapter: 0.156 (a 0.0014 drop, i.e. flat)

**Why it stays flat:** On long free-form IT answers, token overlap with a *single* reference is a weak signal. Two correct troubleshooting answers can use entirely different words. So token-F1 lands in the same ~0.15-0.16 band before and after fine-tuning, and the small per-category swings are mostly noise across 7-8 examples per topic.

**The takeaway:** Token-F1 is the wrong lens here. The chapter uses **format-adherence checks** (does the answer follow the expected IT-support structure?) and an **LLM judge** for the real quality signal. Use those, not token-F1, to decide whether the adapter improved.

---

### 3. Safety Refusal Rate

**What it measures:** Percentage of harmful/unsafe prompts where the model refuses to answer.

**What you see here:**
- Base: 100% (Qwen3-4B-Instruct is well-aligned out of the box)
- Adapter: 60% (-40% REGRESSION)

**Verdict:** This is a safety regression, and it is the single most important result in the report.

**Why this happened:**
- The IT support training data is **helpful-only**: it contains no refusals and no safety examples.
- Fine-tuning on helpful-only data erodes the base model's learned refusal behavior.
- 450 examples of "be helpful and answer the IT question" dilute the alignment that produced refusals.

**How to fix:**
1. **Add safety examples to the training data** (harmful prompts paired with refusals, ~10-20% of the mix). This is the real fix.
2. Keep a refusal slice in the data permanently so retraining doesn't undo it.
3. Re-run the eval and confirm the refusal rate recovers before deploying.

Note that QLoRA in this project's runs kept the refusal rate at 100% on the same eval, so the regression is not inevitable; it depends on the run and the data mix. Always measure it.

---

## Per-Category Results

```
**Per-Category Improvements (LoRA vs base):**
- general: F1 Δ=+0.0172
- hardware: F1 Δ=+0.0112
- networking: F1 Δ=+0.0299
- software: F1 Δ=+0.0138
- linux: F1 Δ=-0.0644
- windows: F1 Δ=-0.0137
- security: F1 Δ=-0.0063
```

**Interpretation:** Each category has only 7-8 examples, so these swings are small and mostly noise. A +0.03 here or a -0.06 there does not establish that the adapter is better or worse at networking versus linux. Treat per-category token-F1 as a sanity check, not a verdict.

---

## Overall Assessment of These Results

### What the numbers say
1. **Token-F1 is flat** (0.158 -> 0.156). Fine-tuning did not move the token-overlap metric, which is expected for long generative IT answers.
2. **Exact match is 0%** throughout, because the model paraphrases.
3. **Safety regressed** from 100% to 60%, because the training data has no refusals.

### What actually changed (use the right tools)
- The adapter changes answer **style and format**, not token overlap. The inference example (`example_inference_base_vs_adapter.md`) shows the base model emitting long, emoji-and-markdown answers while the adapter emits concise IT-support prose. That difference is real and is what fine-tuning bought you; token-F1 just doesn't capture it.
- Use **format-adherence** and an **LLM judge** to measure the quality change, and watch the **safety refusal rate** for regressions.

### Overall verdict

**This is a typical, honest first LoRA run on a small domain dataset.** It teaches two real lessons: (1) token-F1 is the wrong metric for long generative answers, and (2) fine-tuning on helpful-only data can erode safety refusals. Both are worth showing readers rather than hiding behind a cherry-picked metric.

---

## Next Steps to Improve

### Option 1: Restore Safety (the real fix)
1. Collect 50-100 safety examples (harmful prompts with refusal responses).
2. Mix them into the training data (10-20% of total).
3. Retrain with the same hyperparameters and re-check the refusal rate.

### Option 2: Measure Quality the Right Way
- Run the format-adherence check and the LLM judge from the chapter instead of leaning on token-F1.
- Compare base vs adapter on answer structure and tone, as in the inference example.

### Option 3: More Training Data
- Scale the IT support dataset up beyond 450 examples for a stronger, more consistent adapter.
- Keep a permanent safety slice in the mix so retraining never reintroduces the regression.

---

## Key Takeaways

1. **Pick the right metric.** Token-F1 and exact match barely move on long generative IT answers. Use format-adherence and an LLM judge.

2. **Safety matters most here.** The 100% -> 60% refusal drop is the headline result and a deployment blocker until fixed.

3. **Safety data is absent from helpful-only training.** The IT support set has no refusals, which is exactly why refusals eroded. Add them back.

4. **Per-category swings are noise.** With 7-8 examples per topic, don't over-read small deltas.

5. **These are real results.** Don't cherry-pick. Show readers both the flat token-F1 and the safety regression.

---

## For the Chapter

**Suggested narrative:**

> "After training, we evaluate the adapter on 50 held-out IT support examples. The report shows:
>
> - **Token-F1 stayed flat** (0.158 -> 0.156). On long generative answers, token overlap with a single reference is a weak signal, so we rely on format-adherence checks and an LLM judge for the real quality picture.
> - **Exact match is 0%** throughout, because the model paraphrases rather than copying references.
> - **Safety refusal rate dropped from 100% to 60%** (-40%). Our IT support data is helpful-only, with no refusals, so fine-tuning eroded the base model's safety behavior.
>
> The lesson is twofold: choose a metric that fits long generative answers, and watch safety closely. To restore refusals, we'd add explicit safety examples to the training mix and re-evaluate before deployment."

---

## Questions?

**Q: Why is exact match 0%?**  
A: Instruction-tuned models paraphrase long IT answers. Exact match is the wrong tool here; use format-adherence and an LLM judge.

**Q: Token-F1 barely moved. Did fine-tuning do anything?**  
A: Yes, but not in token overlap. It changed answer style and format (concise IT-support prose vs long markdown). Token-F1 doesn't capture that; the inference example does.

**Q: Should I deploy this adapter?**  
A: Not as-is. The safety regression (100% -> 60%) is a blocker. Add safety data and re-check first.

**Q: How do I improve results?**  
A: (1) Add safety examples to training, (2) measure quality with format-adherence and an LLM judge, (3) scale up the dataset while keeping a permanent safety slice.
