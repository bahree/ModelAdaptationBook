#!/bin/bash
# Runtime re-measure for chapter 8 (2026-09-19): full-parameter DPO sharded across three A30s, then LoRA-DPO on one A30.
cd /home/amit/FTBook-pvt/code
export WANDB_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python -m chapter08.train_dpo --model_dir chapter06/runs/sft_run1 \
  --train chapter08/data/preference_pairs/train.jsonl --valid chapter08/data/preference_pairs/valid.jsonl \
  --out chapter08/runs/dpo_rerun_2026-09-19 > chapter08/eval/runtime_2026-09-19/dpo_full_train.log 2>&1
echo "FULL_DONE exit=$?" >> chapter08/eval/runtime_2026-09-19/dpo_full_train.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m chapter08.train_dpo --lora --model_dir chapter06/runs/sft_run1 \
  --train chapter08/data/preference_pairs/train.jsonl --valid chapter08/data/preference_pairs/valid.jsonl \
  --out chapter08/runs/dpo_lora_rerun_2026-09-19 > chapter08/eval/runtime_2026-09-19/dpo_lora_train.log 2>&1
echo "LORA_DONE exit=$?" >> chapter08/eval/runtime_2026-09-19/dpo_lora_train.log
