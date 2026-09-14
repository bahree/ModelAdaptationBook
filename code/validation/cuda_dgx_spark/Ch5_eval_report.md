# Chapter 5 Evaluation Report

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- System prompt: `You are a helpful assistant.`
- Dolly test set: `chapter05/data/smoke/valid.jsonl`
- Adapter: `chapter05/runs/validate_lora_smoke`

## base
### Dolly Test Set (Instruction-Following)
- **Overall exact match**: 100.0%
- **Overall token-F1**: 1.000
- **Test examples**: 1

**Per-Category Accuracy:**
- unknown: EM=100.0%, F1=1.000 (n=1)

- **Safety refusal rate**: 100.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.148

## adapter
### Dolly Test Set (Instruction-Following)
- **Overall exact match**: 100.0%
- **Overall token-F1**: 1.000
- **Test examples**: 1

**Per-Category Accuracy:**
- unknown: EM=100.0%, F1=1.000 (n=1)

- **Safety refusal rate**: 100.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.152

## adapter (Improvement vs Base)
### Dolly Test Set Improvements
- **Overall exact match Δ**: +0.0%
- **Overall token-F1 Δ**: +0.0000

**Per-Category Improvements:**
- unknown: EM Δ=+0.0%, F1 Δ=+0.0000

- **Safety refusal rate Δ**: +0.0%
- **Toy exact match Δ**: +0.0%
- **Toy token-F1 Δ**: +0.0041

