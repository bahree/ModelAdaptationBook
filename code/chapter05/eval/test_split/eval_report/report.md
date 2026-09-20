# Chapter 5 Evaluation Report

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- System prompt: `You are a helpful assistant.`
- Test set: `data/it_support/test.jsonl`
- Adapter: `chapter05/runs/it_lora`

## base
### Test Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.149
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.141 (n=8)
- hardware: EM=0.0%, F1=0.152 (n=7)
- linux: EM=0.0%, F1=0.129 (n=7)
- networking: EM=0.0%, F1=0.169 (n=7)
- security: EM=0.0%, F1=0.199 (n=7)
- software: EM=0.0%, F1=0.122 (n=7)
- windows: EM=0.0%, F1=0.133 (n=7)

- **Safety refusal rate**: 100.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.170

## adapter
### Test Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.154
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.140 (n=8)
- hardware: EM=0.0%, F1=0.188 (n=7)
- linux: EM=0.0%, F1=0.144 (n=7)
- networking: EM=0.0%, F1=0.170 (n=7)
- security: EM=0.0%, F1=0.185 (n=7)
- software: EM=0.0%, F1=0.111 (n=7)
- windows: EM=0.0%, F1=0.141 (n=7)

- **Safety refusal rate**: 60.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.191

## adapter (Improvement vs Base)
### Test Set Improvements
- **Overall exact match Δ**: +0.0%
- **Overall token-F1 Δ**: +0.0048

**Per-Category Improvements:**
- general: EM Δ=+0.0%, F1 Δ=-0.0008
- hardware: EM Δ=+0.0%, F1 Δ=+0.0366
- linux: EM Δ=+0.0%, F1 Δ=+0.0143
- networking: EM Δ=+0.0%, F1 Δ=+0.0012
- security: EM Δ=+0.0%, F1 Δ=-0.0140
- software: EM Δ=+0.0%, F1 Δ=-0.0109
- windows: EM Δ=+0.0%, F1 Δ=+0.0077

- **Safety refusal rate Δ**: -40.0%
- **Toy exact match Δ**: +0.0%
- **Toy token-F1 Δ**: +0.0211

