# Example: Evaluation Report (Base vs LoRA vs QLoRA)

This is a sample output of `chapter05/runs/eval_lora_qlora/report.md` when you compare the **base model**, **LoRA adapter** (`it_lora`), and **QLoRA adapter** (`it_qlora`) using the same IT support eval set. It is produced by running the Step 3 (or Step 5 comparison) evaluation command.

---

## Full report (raw)

Below is the full `report.md` from a run comparing base, LoRA, and QLoRA. Your own report will look like this (paths and numbers may differ slightly).

<details>
<summary>Click to expand full report</summary>

# Chapter 5 Evaluation Report

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- System prompt: `You are a helpful assistant.`
- Eval set: `data/it_support/valid.jsonl`
- Adapter: `chapter05/runs/it_lora`
- Adapter (alt): `chapter05/runs/it_qlora`

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

## adapter_alt
### IT Support Eval Set (Instruction-Following)
- **Overall exact match**: 0.0%
- **Overall token-F1**: 0.163
- **Test examples**: 50

**Per-Category Accuracy:**
- general: EM=0.0%, F1=0.165 (n=8)
- hardware: EM=0.0%, F1=0.229 (n=7)
- linux: EM=0.0%, F1=0.156 (n=7)
- networking: EM=0.0%, F1=0.173 (n=7)
- security: EM=0.0%, F1=0.134 (n=7)
- software: EM=0.0%, F1=0.130 (n=7)
- windows: EM=0.0%, F1=0.152 (n=7)

- **Safety refusal rate**: 100.0%
- **Toy exact match**: 0.0%
- **Toy token-F1**: 0.219

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

## adapter_alt (Improvement vs Base)
### IT Support Eval Set Improvements
- **Overall exact match Δ**: +0.0%
- **Overall token-F1 Δ**: +0.0049

**Per-Category Improvements:**
- general: EM Δ=+0.0%, F1 Δ=+0.0247
- hardware: EM Δ=+0.0%, F1 Δ=+0.0315
- linux: EM Δ=+0.0%, F1 Δ=-0.0418
- networking: EM Δ=+0.0%, F1 Δ=+0.0253
- security: EM Δ=+0.0%, F1 Δ=+0.0135
- software: EM Δ=+0.0%, F1 Δ=-0.0111
- windows: EM Δ=+0.0%, F1 Δ=-0.0106

- **Safety refusal rate Δ**: +0.0%
- **Toy exact match Δ**: +0.0%
- **Toy token-F1 Δ**: +0.0491

</details>

---

## How to read this report

### Sections

| Section | Meaning |
|--------|--------|
| **base** | Base model only (Qwen3-4B-Instruct). No adapter. |
| **adapter** | Base + LoRA adapter (`it_lora`). |
| **adapter_alt** | Base + QLoRA adapter (`it_qlora`). |
| **adapter (Improvement vs Base)** | LoRA vs base: deltas (Δ) for each metric. |
| **adapter_alt (Improvement vs Base)** | QLoRA vs base: deltas (Δ) for each metric. |

### Metrics

| Metric | What it is | What to look for |
|--------|------------|-------------------|
| **Overall exact match (EM)** | % of answers that exactly match the reference (after normalization). | Almost always 0% here: instruction-tuned models paraphrase long IT answers rather than copy them. Don't read into it. |
| **Overall token-F1** | Token-level F1 (overlap with reference). | On long free-form answers this is a weak signal; expect it to stay roughly flat across base, LoRA, and QLoRA. |
| **Per-category (e.g. networking, hardware)** | Same metrics broken down by IT topic. | Small per-category swings on 7-8 examples each are mostly noise. |
| **Safety refusal rate** | % of harmful prompts the model refused. | The signal that matters. Base is 100%. Fine-tuning on helpful-only data can lower it. |
| **Toy exact match / Toy token-F1** | Same metrics on a small toy set. | Sanity check; small sample so can be noisy. |

### What this example shows

- **Task performance (token-F1) is essentially flat:**  
  Base 0.158, LoRA 0.156, QLoRA 0.163. All three land in the same ~0.15-0.16 band. Token-F1 is simply the wrong lens for long generative IT answers, where two correct answers can share few tokens with the single reference. The chapter relies on format-adherence checks and an LLM judge for the real quality signal.

- **Safety is the real story:**  
  Base 100% -> LoRA 60% (−40%) -> QLoRA 100% (no change). The LoRA adapter shows a clear safety regression because the IT support training data is helpful-only (no refusals). **QLoRA kept refusals at 100%** in this run and its token-F1 edged slightly above base, which is worth stating plainly.

- **Per-category:**  
  Differences are small (each category has only 7-8 examples), so treat per-category swings as noise rather than signal.

- **Exact match:**  
  0% for all three is normal; the model paraphrases rather than copying references.

**Bottom line:** Token-F1 barely separates the variants, so the decision rests elsewhere. Watch the safety refusal rate (LoRA regressed, QLoRA held) and use the format-adherence and LLM-judge checks from the chapter to assess answer quality. If you adopt the LoRA adapter, add explicit refusal examples to the training mix first.
