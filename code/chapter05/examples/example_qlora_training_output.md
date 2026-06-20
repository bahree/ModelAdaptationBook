# Example: QLoRA Training Output

This file captures a typical run of `train_qlora` for the Chapter 5 IT support dataset (450 train, 50 valid, 3 epochs). Use it to recognize normal output and interpret the metrics.

## Command

```bash
python -m chapter05.train_qlora \
  --train data/it_support_fmt/train.jsonl \
  --valid data/it_support/valid.jsonl \
  --out chapter05/runs/it_qlora
```

## Raw output

```
Loading checkpoint shards: 100%|██████████| 3/3 [00:09<00:00,  3.32s/it]
Tokenizing train dataset: 100%|██████████| 450/450 [00:00<00:00, 1481.83 examples/s]
Tokenizing eval dataset: 100%|██████████| 50/50 [00:00<00:00, 1420.73 examples/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': 2.4886, 'grad_norm': 0.4785, 'learning_rate': 0.0001895, 'entropy': 1.627, 'num_tokens': 19854.0, 'mean_token_accuracy': 0.5246, 'epoch': 0.18}
{'loss': 1.938, 'grad_norm': 0.6562, 'learning_rate': 0.0001778, 'entropy': 1.951, 'num_tokens': 37793.0, 'mean_token_accuracy': 0.5871, 'epoch': 0.36}
{'loss': 1.8182, 'grad_norm': 0.5937, 'learning_rate': 0.0001661, 'entropy': 1.810, 'num_tokens': 57150.0, 'mean_token_accuracy': 0.6004, 'epoch': 0.53}
{'loss': 1.7146, 'grad_norm': 0.3809, 'learning_rate': 0.0001544, 'entropy': 1.644, 'num_tokens': 74968.0, 'mean_token_accuracy': 0.6237, 'epoch': 0.71}
{'loss': 1.73, 'grad_norm': 0.4180, 'learning_rate': 0.0001427, 'entropy': 1.672, 'num_tokens': 93889.0, 'mean_token_accuracy': 0.6102, 'epoch': 0.89}
{'eval_loss': 2.1721, 'eval_runtime': 6.9667, 'eval_samples_per_second': 7.177, 'eval_entropy': 2.1441, 'eval_mean_token_accuracy': 0.5311, 'epoch': 0.89}
{'loss': 1.7382, 'grad_norm': 0.4082, 'learning_rate': 0.0001310, 'entropy': 1.708, 'num_tokens': 111381.0, 'mean_token_accuracy': 0.6172, 'epoch': 1.05}
{'loss': 1.6778, 'grad_norm': 0.3652, 'learning_rate': 0.0001193, 'entropy': 1.659, 'num_tokens': 130462.0, 'mean_token_accuracy': 0.6277, 'epoch': 1.23}
{'loss': 1.6193, 'grad_norm': 0.4570, 'learning_rate': 0.0001076, 'entropy': 1.595, 'num_tokens': 150737.0, 'mean_token_accuracy': 0.6324, 'epoch': 1.41}
{'loss': 1.7195, 'grad_norm': 0.5039, 'learning_rate': 9.591e-05, 'entropy': 1.647, 'num_tokens': 168985.0, 'mean_token_accuracy': 0.6204, 'epoch': 1.59}
{'loss': 1.6138, 'grad_norm': 0.5703, 'learning_rate': 8.421e-05, 'entropy': 1.588, 'num_tokens': 187364.0, 'mean_token_accuracy': 0.6333, 'epoch': 1.76}
{'eval_loss': 2.1688, 'eval_runtime': 7.077, 'eval_samples_per_second': 7.065, 'eval_entropy': 2.0761, 'eval_mean_token_accuracy': 0.5320, 'epoch': 1.76}
{'loss': 1.6855, 'grad_norm': 0.5391, 'learning_rate': 7.251e-05, 'entropy': 1.603, 'num_tokens': 204326.0, 'mean_token_accuracy': 0.6337, 'epoch': 1.94}
{'loss': 1.6758, 'grad_norm': 0.5469, 'learning_rate': 6.082e-05, 'entropy': 1.652, 'num_tokens': 222326.0, 'mean_token_accuracy': 0.6291, 'epoch': 2.11}
{'loss': 1.5646, 'grad_norm': 0.5508, 'learning_rate': 4.912e-05, 'entropy': 1.580, 'num_tokens': 240777.0, 'mean_token_accuracy': 0.6366, 'epoch': 2.28}
{'loss': 1.5365, 'grad_norm': 0.5898, 'learning_rate': 3.743e-05, 'entropy': 1.479, 'num_tokens': 260167.0, 'mean_token_accuracy': 0.6551, 'epoch': 2.46}
{'loss': 1.5394, 'grad_norm': 0.6406, 'learning_rate': 2.573e-05, 'entropy': 1.478, 'num_tokens': 277801.0, 'mean_token_accuracy': 0.6528, 'epoch': 2.64}
{'eval_loss': 2.1885, 'eval_runtime': 6.9719, 'eval_samples_per_second': 7.172, 'eval_entropy': 1.9939, 'eval_mean_token_accuracy': 0.5302, 'epoch': 2.64}
{'loss': 1.5534, 'grad_norm': 0.6719, 'learning_rate': 1.404e-05, 'entropy': 1.505, 'num_tokens': 296656.0, 'mean_token_accuracy': 0.6424, 'epoch': 2.82}
{'loss': 1.5505, 'grad_norm': 0.6797, 'learning_rate': 2.339e-06, 'entropy': 1.522, 'num_tokens': 315417.0, 'mean_token_accuracy': 0.6514, 'epoch': 3.0}
{'train_runtime': 852.5782, 'train_samples_per_second': 1.583, 'train_steps_per_second': 0.201, 'train_loss': 1.7152, 'mean_token_accuracy': 0.6443, 'epoch': 3.0}
100%|██████████| 171/171 [14:12<00:00,  4.99s/it]
Saved QLoRA adapter to: chapter05/runs/it_qlora
```

## What this means

| Output | Meaning |
|--------|---------|
| **Loading checkpoint shards (3/3)** | The base model is loaded in **4-bit** (quantized) across its three weight shards. Quantization is what makes QLoRA fit in a small GPU. |
| **Tokenizing (450 train / 50 eval)** | Train and eval datasets are converted to token IDs. The IT support dataset has 450 training rows and 50 held-out validation rows. |
| **Tokenizer PAD/BOS/EOS message** | Qwen has no explicit PAD token; the trainer sets `pad_token_id` (to EOS, 151643) and updates the config. **Expected and harmless**: training and generation work correctly. |
| **loss** | Training loss (cross-entropy). It **decreases** over the three epochs (2.49 -> ~1.55). Small ups and downs between logging steps are normal. |
| **grad_norm** | Gradient norm; reflects update size. Stable values (here ~0.4-0.7) indicate healthy training. |
| **learning_rate** | Current learning rate (the scheduler decays it over time). Starts near 2e-4 and winds down to ~2e-6 by the end of epoch 3. |
| **mean_token_accuracy** | Fraction of next-token predictions matching the target. A **rough proxy for quality**; rising (0.52 -> 0.64) suggests the model is learning. |
| **eval_loss / eval_mean_token_accuracy** | Validation metrics logged three times across training. Eval loss holds around 2.17-2.19 (this short run plateaus quickly on 450 examples); used to watch for overfitting. |
| **train_runtime: 852.6** | Total training time in **seconds** (~14.2 minutes). The progress bar (171 steps, ~5.0 s/step) confirms the pace. |
| **Saved QLoRA adapter to: ...** | Adapter weights written to `chapter05/runs/it_qlora`. Use this path for evaluation (Step 3) and inference with `--adapter` and `--quantized_4bit`. |

**Summary:** Training loss and token accuracy improve over 3 epochs, validation tracks training (eval loss plateaus, which is expected with only 450 examples), and the run finishes by saving the adapter. The tokenizer message is safe to ignore for local runs.

## Screenshots

Training progress and GPU usage from a typical QLoRA run:

![QLoRA training](../images/chap5-qlora_training.png)

![QLoRA training (GPU)](../images/chap5-qlora_training_gpu.png)
