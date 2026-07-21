"""Chat template encoding for supervised fine-tuning with assistant-only loss masking.

Provides helpers to:
    - Format system + user + assistant messages into the model's chat template.
    - Encode messages into input_ids/labels with prompt tokens masked (-100)
      so the loss is computed only on the assistant's response tokens.
    - Build generation prompts for inference.

These utilities are used by the training scripts (train_lora.py, train_qlora.py),
evaluation (eval.py), and inference (generate.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def ensure_padding(tokenizer) -> None:
    """Set up padding for the tokenizer if not already configured.

    Many models (including Qwen) ship without an explicit pad token. This
    function assigns EOS as the pad token and sets right-padding, which is
    required for batch processing during both training and evaluation.

    Args:
        tokenizer: A HuggingFace tokenizer instance (modified in place).
    """
    # Qwen tokenizers often ship without an explicit pad token. Reuse EOS for padding.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


@dataclass(frozen=True)
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def encode_chat_for_sft(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    max_length: int,
) -> EncodedExample:
    """Encode chat messages for supervised fine-tuning with assistant-only loss.

    We compute labels for the assistant response only by masking tokens that belong to:
    - system message
    - user message(s)
    - assistant role/header tokens (generation prompt)
    """
    ensure_padding(tokenizer)

    # Prefix: everything up to the assistant "start" token/header.
    prefix_messages = [m for m in messages if m["role"] != "assistant"]
    if not prefix_messages:
        raise ValueError("messages must include at least a user/system message")

    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = list(full["input_ids"])
    attention_mask = list(full.get("attention_mask") or [1] * len(input_ids))

    labels = list(input_ids)
    prefix_len = len(prefix["input_ids"])
    if prefix_len > len(labels):
        prefix_len = len(labels)
    for i in range(prefix_len):
        labels[i] = -100

    return EncodedExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build a generation prompt string from system+user messages."""
    ensure_padding(tokenizer)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

