# Chapter 5 Evaluation Report: r=16 vs r=8 Comparison

**Purpose:** Compare LoRA rank 16 vs rank 8 to test whether a smaller rank preserves safety.

**Result:** Dropping the rank to r=8 helped neither task performance nor safety.

---

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- System prompt: `You are a helpful assistant.`
- Eval set: `data/it_support/valid.jsonl`
- Adapter: `chapter05/runs/it_lora` (r=16)
- Adapter (alt): `chapter05/runs/it_lora_r8` (r=8)

## base
### IT Support Eval Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.158
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.140 (n=8)
- hardware: EM=0.0%, F1=0.198 (n=7)
- linux: EM=0.0%, F1=0.198 (n=7)
- networking: EM=0.0%, F1=0.148 (n=7)
- security: EM=0.0%, F1=0.120 (n=7)
- software: EM=0.0%, F1=0.141 (n=7)
- windows: EM=0.0%, F1=0.162 (n=7)

- **Safety refusal rate**: 100.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.170

## adapter (r=16)
### IT Support Eval Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.156
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.158 (n=8)
- hardware: EM=0.0%, F1=0.209 (n=7)
- linux: EM=0.0%, F1=0.133 (n=7)
- networking: EM=0.0%, F1=0.178 (n=7)
- security: EM=0.0%, F1=0.114 (n=7)
- software: EM=0.0%, F1=0.155 (n=7)
- windows: EM=0.0%, F1=0.149 (n=7)

- **Safety refusal rate**: 60.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.191

## adapter_alt (r=8)
### IT Support Eval Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.146
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.139 (n=8)
- hardware: EM=0.0%, F1=0.212 (n=7)
- linux: EM=0.0%, F1=0.144 (n=7)
- networking: EM=0.0%, F1=0.147 (n=7)
- security: EM=0.0%, F1=0.117 (n=7)
- software: EM=0.0%, F1=0.114 (n=7)
- windows: EM=0.0%, F1=0.149 (n=7)

- **Safety refusal rate**: 60.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.241

## adapter (r=16) vs Base
### IT Support Eval Set Improvements
- **Overall exact match Δ**: +0.0%
- **Overall token-F1 Δ**: -0.0014

**Per-Category Improvements:**
- general: EM Δ=+0.0%, F1 Δ=+0.0172
- hardware: EM Δ=+0.0%, F1 Δ=+0.0112
- linux: EM Δ=+0.0%, F1 Δ=-0.0644
- networking: EM Δ=+0.0%, F1 Δ=+0.0299
- security: EM Δ=+0.0%, F1 Δ=-0.0063
- software: EM Δ=+0.0%, F1 Δ=+0.0138
- windows: EM Δ=+0.0%, F1 Δ=-0.0137

- **Safety refusal rate Δ**: -40.0%
- **Toy exact match Δ**: +0.0%
- **Toy token-F1 Δ**: +0.0211

## adapter_alt (r=8) vs Base
### IT Support Eval Set Improvements
- **Overall exact match Δ**: +0.0%
- **Overall token-F1 Δ**: -0.0118

**Per-Category Improvements:**
- general: EM Δ=+0.0%, F1 Δ=-0.0010
- hardware: EM Δ=+0.0%, F1 Δ=+0.0145
- linux: EM Δ=+0.0%, F1 Δ=-0.0532
- networking: EM Δ=+0.0%, F1 Δ=-0.0013
- security: EM Δ=+0.0%, F1 Δ=-0.0036
- software: EM Δ=+0.0%, F1 Δ=-0.0270
- windows: EM Δ=+0.0%, F1 Δ=-0.0130

- **Safety refusal rate Δ**: -40.0%
- **Toy exact match Δ**: +0.0%
- **Toy token-F1 Δ**: +0.0704

---

## Analysis: Why r=8 Didn't Help

### What we expected
- r=8 would preserve more base model behavior
- Safety refusal rate would recover (60% -> higher)
- Task performance might dip slightly

### What actually happened
- Token-F1 went the wrong way: r=16 was 0.156, r=8 was 0.146 (both below base 0.158).
- Safety did not recover: r=8 stayed at 60%, the same regression as r=16.
- Per-category swings are small and mostly within noise for 7-8 examples per topic.

### Why
1. **No safety data** — Neither run had refusal examples in training, so neither could relearn refusals.
2. **Rank reduction can't fix a data problem** — The issue is dataset composition (helpful-only), not model capacity.
3. **Token-F1 is the wrong lens** — On long generative IT answers, token overlap with a single reference barely separates the variants. Use format-adherence checks and an LLM judge for the real quality signal.
4. **Small eval set variance** — 50 examples, with five safety prompts, means individual swings are noisy.

### Key lesson
**You can't parameter-tune your way out of a data problem.** To restore safety, add explicit refusal examples to the training data.

### The real fix
1. Add safety examples (harmful prompts paired with refusals).
2. Make them 10-20% of the training mix.
3. Retrain and re-evaluate the refusal rate.
4. Only then can you trust the model's safety behavior.

---

## Pedagogical value

This null result is **more instructive** than a tidy win, because it teaches:

1. **Validate empirically** — Don't assume a smaller rank will preserve safety.
2. **Data trumps parameters** — Fix data problems with data, not hyperparameters.
3. **Pick the right metric** — Token-F1 hides the real story on long generative answers.
4. **Safety needs explicit training** — It won't re-emerge from rank tuning.

This is an honest look at fine-tuning that helps readers avoid the same mistake.
