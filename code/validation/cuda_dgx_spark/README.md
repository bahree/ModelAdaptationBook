# DGX Spark Validation (Chapters 1-5)

This folder captures a real reader-style validation run on DGX Spark class hardware (NVIDIA GB10).

## Environment

See [./_env.log](./_env.log).

Summary:
- torch 2.13.0+cu130
- CUDA available: True
- GPU: NVIDIA GB10
- Platform: Linux aarch64

## What Was Run (Chapter by Chapter)

### Chapter 1
- PASS: base-only sidebar run
  - [Ch1_sidebar__base-only_.log](./Ch1_sidebar__base-only_.log)
- PASS: base + LoRA adapter run (using Chapter 5 smoke adapter)
  - [Ch1_sidebar__base_plus_lora_.log](./Ch1_sidebar__base_plus_lora_.log)
- Expected skip: SFT path skipped because Chapter 6 artifact was not present in this chapter-1-to-5 scope.

### Chapter 2
- PASS: adapter preview script (resolved Hub adapter and generated outputs)
  - [Ch2_adapter_preview.log](./Ch2_adapter_preview.log)
- PASS: full quickstart LoRA training run
  - [Ch2_quickstart__LoRA_.log](./Ch2_quickstart__LoRA_.log)

### Chapter 3
- PASS: dataset manifest runnable module
  - [Ch3_dataset_manifest.log](./Ch3_dataset_manifest.log)
- PASS: end-to-end synthetic data generation pipeline (teacher calls + quality gate + save)
  - [Ch3_synthetic_pipeline.log](./Ch3_synthetic_pipeline.log)
- Note: the heavier data-quality experiment script (4 training conditions) was not executed in this run because it is a long benchmark-style workload; dependencies import correctly.

### Chapter 4
- PASS: few-shot mock backend
  - [Ch4_few_shot_mock.log](./Ch4_few_shot_mock.log)
- PASS: few-shot HF backend on GPU
  - [Ch4_few_shot_hf.log](./Ch4_few_shot_hf.log)
- PASS: prompt validator mock backend
  - [Ch4_prompt_validator_mock.log](./Ch4_prompt_validator_mock.log)
- PASS: RAG retrieval evaluation (hash backend)
  - [Ch4_rag_eval_hash.log](./Ch4_rag_eval_hash.log)

### Chapter 5
- PASS: chapter validator with tiny LoRA smoke training on CUDA
  - Captured in terminal output during validation run
- PASS: inference generation using smoke adapter
  - [Ch5_generate__load_adapter_.log](./Ch5_generate__load_adapter_.log)
- PASS: base vs adapter smoke evaluation pipeline
  - [Ch5_evaluate__base_vs_smoke_adapter_.log](./Ch5_evaluate__base_vs_smoke_adapter_.log)
  - [Ch5_eval_report.md](./Ch5_eval_report.md)

## Issues Encountered and Fix/Recommendation

1. High severity: wrong CUDA wheel for GB10 (sm_121)
- Symptom: torch 2.13.0+cu126 detected CUDA but warned kernels were not built for compute capability 12.1.
- Impact: training/inference can fail or silently use unsupported paths.
- Fix used in this validation: install torch wheels from cu130 index.
- Recommendation for docs/recipes: on DGX Spark/GB10, default to cu130 or newer wheels.

2. Medium severity: deprecation warning from transformers keyword
- Symptom: torch_dtype warning asking to use dtype.
- Where observed: Chapter 1 and Chapter 4 HF run logs.
- Impact: no runtime failure, but noisy and future compatibility risk.
- Recommendation: migrate from torch_dtype to dtype in model loading calls.

3. Medium severity: generation config warnings (temperature/top_p/top_k)
- Symptom: warning that generation flags may be ignored.
- Where observed: Chapter 1, Chapter 2 adapter preview, Chapter 4 HF, Chapter 5 generate/eval.
- Impact: behavior still runs, but warning can confuse readers.
- Recommendation: align generation config fields with current transformers version and remove stale flags.

4. Medium severity: chapter scope coupling
- Symptom: Chapter 1 script optionally expects Chapter 6 SFT artifact.
- Impact: readers doing chapters 1-5 only will see expected skip message.
- Recommendation: keep current behavior but document this explicitly in chapter 1 quick-start notes.

## Overall Verdict (Reader on DGX Spark)

- Chapters 1-5 are runnable on DGX Spark with GPU acceleration when torch is installed from a GB10-compatible CUDA index (cu130 tested here).
- All executed chapter paths in this validation passed.
- The primary blocker was environment setup (GPU architecture wheel mismatch), not code logic.
