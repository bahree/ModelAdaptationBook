"""Dataset preparation for Chapter 5 training and evaluation.

- prepare_dataset_for_sft(): turns ChatExamples into rows with 'messages' for SFTTrainer.
- encode_examples(): turns ChatExamples into input_ids/labels for loss eval (e.g. eval_loss_on_jsonl).
- EncodedChatDataset: PyTorch Dataset wrapper.
Used by train_lora/train_qlora (SFTTrainer) and eval.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from .chat_template import encode_chat_for_sft
from .data import ChatExample


class EncodedChatDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        # Extract column names from the first row (required by TRL's SFTTrainer)
        self._column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._rows[idx]
    
    @property
    def column_names(self) -> List[str]:
        """Return column names (required by TRL's SFTTrainer)."""
        return self._column_names


def encode_examples(
    tokenizer,
    examples: Sequence[ChatExample],
    *,
    max_length: int,
) -> EncodedChatDataset:
    """Encode examples for use with standard Trainer (e.g., for evaluation).
    
    For training with SFTTrainer, use prepare_dataset_for_sft() instead.
    """
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        enc = encode_chat_for_sft(tokenizer, ex.messages, max_length=max_length)
        rows.append(
            {
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
                "labels": enc.labels,
            }
        )
    return EncodedChatDataset(rows)


def prepare_dataset_for_sft(
    examples: Sequence[ChatExample],
) -> HFDataset:
    """Prepare dataset for SFTTrainer using messages format.
    
    SFTTrainer will automatically apply the chat template and handle loss masking.
    Returns a Hugging Face Dataset (required by TRL's SFTTrainer).
    """
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        rows.append({"messages": ex.messages})
    
    # Convert to Hugging Face Dataset (required by TRL's SFTTrainer)
    return HFDataset.from_list(rows)

