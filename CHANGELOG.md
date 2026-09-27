# Changelog

All notable changes to the code in this repository are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses calendar-versioned releases tied to Manning MEAP drops (`MEAP-vN`) and a final `1.0.0` at print publication.

## [Unreleased]

<!-- Changes landing on `main` between releases go here. -->

### Added

- **Chapters 6 through 9** (2026-09-23) — the remaining hands-on code: full supervised fine-tuning (`chapter06`), black-box distillation (`chapter07`), DPO with GRPO/RFT examples (`chapter08`), and the operations toolkit (`chapter09`: JSON model registry, TF-IDF drift detector, rollback demo, safety monitor). All run on the shared IT-support dataset in `data/it_support/` with the held-out test split (`--split test`).

### Changed

- **Chapter 6** (2026-09-26) — `eval/safety/` now holds the run the chapter prints (base 62%, fine-tuned 62%, harmful-request refusal 100% to 50%, regression flagged); the README's safety summary matches it.
- **Chapter 8** (2026-09-26) — README training-time row reflects the measured runs (about 3.5 min full DPO on three 24 GB cards, about 2.8 min LoRA-DPO on one).
- **Chapter 9** (2026-09-26) — README safety-monitor compare command writes `safety_report_dpo.json` against `safety_report.json`; `model_registry.py` docstring shows `--registry_dir` before the subcommand; the retired `data/drift_report_same_domain.json` removed.

## [MEAP-v0.1] — TBD

First public release of the code repository alongside Manning's MEAP launch.

### Added

- Initial release of code for Chapters 1 through 5.
- **Chapter 1** — reproducibility script for the §1.6 sidebar (`run_sidebar_example.py`). Runs the chapter's prompt through base Qwen3-4B, the Chapter 5 LoRA adapter, and the Chapter 6 SFT model side by side; degrades gracefully when later-chapter artifacts are not yet built.
- **Chapter 2** — five-step LoRA quickstart (`quickstart.py`) on Qwen3-4B-Instruct-2507 with a 40-example slice of the IT-support dataset: prepare data, load the base model with a LoRA config, train 20 steps with TRL's `SFTTrainer`, compare outputs before and after, save the adapter with a manifest. Plus `run_chapter5_adapter.py`, which previews the chapter 5 adapter (local or from the Hub) on the same prompts.
- **Chapter 3** — data-quality experiment, six-step synthetic-data-generation pipeline using a frontier teacher, and a standalone `DatasetManifest` module for content hashing and lineage tracking.
- **Chapter 4** — few-shot ticket classifier, many-shot prompt assembly, prompt validator with run-to-run variability measurement, minimal RAG pipeline (50 lines), Precision@k / Recall@k / Hit@1 retrieval evaluator.
- **Chapter 5** — LoRA and QLoRA training, evaluation, and inference on a 400-example Dolly subset of Qwen3-4B-Instruct-2507; published adapter on Hugging Face Hub at `bahree/qwen3-4b-dolly-lora-ch5`.
- Shared utilities in `common/` (JSONL I/O, env loading, deterministic seeding, manifest tracking, OpenRouter helper).
- CI workflow (`pytest` on Ubuntu and Windows with Python 3.11).
- Issue templates for bug reports and errata; redirect for book-content questions to Manning's liveBook forum.
- Contributing guidelines.

### Notes

- Chapters 6 through 9 (Full SFT, Distillation, DPO, Operations) landed on `main` on 2026-09-23 (see Unreleased above).
- All hands-on chapters use Qwen3-4B-Instruct-2507 as the base model and Databricks Dolly-15K (filtered subsets) as the dataset, so the chapters compose into a single coherent example pipeline.

---

## Release tagging convention

Each MEAP release is tagged in Git so readers can check out the exact tree that matches the manuscript they're reading.

```bash
git tag                                       # list all releases
git checkout MEAP-v0.1                        # check out a specific release
```
