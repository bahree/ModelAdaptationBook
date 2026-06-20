# Example: LoRA vs QLoRA Evaluation Output

This file captures a typical run of the evaluation script when comparing the **base model**, **LoRA adapter** (`it_lora`), and **QLoRA adapter** (`it_qlora`) on the same eval set. Use it to recognize normal output and the order of steps.

## Command

```bash
python chapter05/scripts/listing_5_3_evaluate.py \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/it_lora \
  --adapter_alt chapter05/runs/it_qlora \
  --dolly_test data/it_support/valid.jsonl
```

## Raw output

```
Step 1/4: Loading base model...
Loading checkpoint shards: 100%|██████████| 3/3 [00:02<00:00,  1.32it/s]
✓ Base model loaded

Step 2/4: Evaluating base model...
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
  Evaluating examples... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Evaluating toy test set... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Running safety checks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
✓ Base evaluation complete

Step 3/4: Loading adapter from chapter05/runs/it_lora...
Loading checkpoint shards: 100%|██████████| 3/3 [00:02<00:00,  1.25it/s]
✓ Adapter loaded

Step 4/4: Evaluating fine-tuned model...
  Evaluating examples... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Evaluating toy test set... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Running safety checks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
✓ Fine-tuned evaluation complete

Loading alternative adapter from chapter05/runs/it_qlora...
Loading checkpoint shards: 100%|██████████| 3/3 [00:02<00:00,  1.30it/s]
✓ Alternative adapter loaded

Evaluating alternative adapter...
  Evaluating examples... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Evaluating toy test set... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Running safety checks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
✓ Alternative evaluation complete


Writing evaluation reports...

✓ Evaluation complete!
✓ JSON report: chapter05/runs/eval_lora_qlora/report.json
✓ Markdown summary: chapter05/runs/eval_lora_qlora/report.md

→ View the markdown report for a human-readable summary
```

## What this means

| Output | Meaning |
|--------|---------|
| **Step 1/4: Loading base model** | Base model (Qwen3-4B) is loaded once across its three checkpoint shards. |
| **Step 2/4: Evaluating base model** | The IT support eval set (instruction-following), toy test set, and safety suite run on the base model only. Progress bars show completion. |
| **Step 3/4: Loading adapter** | LoRA adapter (`it_lora`) is attached to the same base. |
| **Step 4/4: Evaluating fine-tuned model** | The same three evals (IT support, toy, safety) run with the LoRA adapter. |
| **Loading alternative adapter** | QLoRA adapter (`it_qlora`) is loaded (base is reloaded and this adapter is attached). |
| **Evaluating alternative adapter** | The same three evals run with the QLoRA adapter. |
| **Writing evaluation reports** | Results are written to `chapter05/runs/eval_lora_qlora/report.json` and `report.md`. |

**Summary:** The script evaluates base -> LoRA -> QLoRA in sequence and writes a single report that compares all three. Open `chapter05/runs/eval_lora_qlora/report.md` for the human-readable summary. For a full example of that report (base vs LoRA vs QLoRA) and how to interpret each section and metric, see [example_eval_report_lora_vs_qlora.md](example_eval_report_lora_vs_qlora.md).

## Screenshot (terminal / report)

![LoRA vs QLoRA evaluation](../images/chap5-qlora_lora_evals.png)
