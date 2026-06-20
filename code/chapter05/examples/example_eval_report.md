# Chapter 5 eval report

This is a sample of the base-vs-LoRA evaluation report produced by `listing_5_3_evaluate.py` on the IT support dataset. Your own numbers will differ slightly from run to run.

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- System prompt: `You are a helpful assistant.`
- Eval set: `data/it_support/valid.jsonl` (50 IT-only examples)
- Adapter: `chapter05/runs/it_lora` (LoRA, r=16)

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

## adapter
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

## adapter (Improvement vs Base)
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

## How to read this

- **Exact match is 0% everywhere.** Instruction-tuned models paraphrase rather than copy the reference answer word for word, so exact match stays at zero on long generative IT answers. That is expected, not a failure.
- **Overall token-F1 barely moves** (base 0.158 -> LoRA 0.156). On long free-form IT answers, token overlap with a single reference is a weak signal: two correct troubleshooting answers can share few tokens. The chapter uses format-adherence checks and an LLM judge for the real quality signal; token-F1 is the wrong lens here.
- **Safety refusal rate drops from 100% to 60%.** This is the result that matters. The IT support training data is helpful-only (it contains no refusals), so fine-tuning on it erodes the base model's refusal behavior. Watch this metric and add explicit refusal examples to the training mix if you need to restore it.
