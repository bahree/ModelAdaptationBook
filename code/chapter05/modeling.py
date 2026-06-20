"""Model loading utilities for LoRA and QLoRA fine-tuning (Listing 5.2).

Provides functions to:
    - Load the base model in full precision (for LoRA) or 4-bit quantized (for QLoRA).
    - Load and configure the tokenizer with proper padding.
    - Create and apply LoRA adapter configurations.

Used by train_lora.py, train_qlora.py, generate.py, and eval.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, prepare_model_for_kbit_training

from common.hub import split_ref, subfolder_kwargs

from .chat_template import ensure_padding


# Standard attention and MLP projection modules in Transformer architectures.
# Adapting all attention projections (q/k/v/o) plus MLP projections (up/gate/down)
# gives the best quality/cost balance. See Section 5.5 for guidance.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
]


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any


def load_tokenizer(model_name: str):
    """Load and configure a tokenizer with proper padding for the given model.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        A configured AutoTokenizer with padding set (see ensure_padding).
    """
    p, s = split_ref(model_name)
    tok = AutoTokenizer.from_pretrained(p, **subfolder_kwargs(s), trust_remote_code=True)
    ensure_padding(tok)
    return tok


def _resolve_device_map(device_map):
    """Resolve "auto" to a concrete placement that avoids CPU/disk OFFLOAD.

    device_map="auto" offloads layers to CPU/disk on memory-constrained or
    non-CUDA machines. That offload path trips a PEFT adapter-loading bug
    (KeyError in _update_offload, e.g.
    'base_model.model.model.model.embed_tokens') when a LoRA adapter is later
    attached for eval/inference, which is exactly what a reader on Apple
    Silicon (MPS) or CPU hits in eval.py / generate.py. Pinning everything to a
    single device builds no offload index, so the adapter loads cleanly. The 4B
    base fits on one MPS/CPU device for inference and 1-epoch LoRA.
    """
    if device_map != "auto":
        return device_map
    if torch.cuda.is_available():
        return "auto"  # a single 24GB+ CUDA card fits the 4B with no offload
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def load_base_model_lora(
    model_name: str,
    *,
    device_map: str = "auto",
    dtype: str | torch.dtype = "auto",
    gradient_checkpointing: bool = True,
):
    """Load the base model in full precision for LoRA fine-tuning or inference.

    Args:
        model_name: HuggingFace model ID or local path.
        device_map: Device placement strategy. "auto" distributes layers across
            available GPUs (or CPU if no GPU), which is the simplest approach
            for single-GPU setups.
        dtype: Weight dtype. "auto" lets HF choose the best dtype for the hardware.
        gradient_checkpointing: If True, trades compute for memory by recomputing
            activations during backward. Roughly halves memory at ~20% speed cost.
            Disable for inference (no backward pass needed).

    Returns:
        A HuggingFace AutoModelForCausalLM ready for LoRA adapter attachment.
    """
    p, s = split_ref(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        p,
        **subfolder_kwargs(s),
        device_map=_resolve_device_map(device_map),
        dtype=dtype,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # KV cache is incompatible with gradient checkpointing during training.
        model.config.use_cache = False
    return model


def load_base_model_qlora(
    model_name: str,
    *,
    device_map: str = "auto",
    compute_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
):
    """Load the base model in 4-bit quantized form for QLoRA fine-tuning.

    Uses bitsandbytes NF4 (NormalFloat4) quantization with double quantization
    to compress the base model to ~4 bits per parameter. This reduces GPU memory
    by roughly 4x compared to full precision, enabling fine-tuning of larger
    models on smaller GPUs.

    Args:
        model_name: HuggingFace model ID or local path.
        device_map: Device placement strategy (same as load_base_model_lora).
        compute_dtype: Dtype for computation during forward/backward passes.
            bf16 is preferred for its wider dynamic range.
        gradient_checkpointing: If True, enable gradient checkpointing for
            additional memory savings. Disable for inference.

    Returns:
        A quantized model prepared for k-bit training (gradients enabled on
        non-quantized parameters like LayerNorm and LoRA adapters).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA (4-bit) requires a CUDA GPU: bitsandbytes does not support "
            "Apple Silicon (MPS) or CPU. On those platforms use LoRA instead "
            "(train_lora / load_base_model_lora), which runs in full precision "
            "and fits the 4B model on a single MPS device."
        )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # NF4 (NormalFloat4): a 4-bit data type optimized for normally-distributed
        # weights, giving higher precision near zero where most weights cluster.
        bnb_4bit_quant_type="nf4",
        # Double quantization: further compresses the quantization constants
        # themselves, saving ~0.4 bits per parameter with minimal quality loss.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    p, s = split_ref(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        p,
        **subfolder_kwargs(s),
        device_map=_resolve_device_map(device_map),
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    # prepare_model_for_kbit_training enables gradients on non-quantized layers
    # (LayerNorm, embeddings) and sets up proper dtype casting for mixed-precision
    # training with quantized weights.
    model = prepare_model_for_kbit_training(model)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def create_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> LoraConfig:
    """Create a LoRA configuration for use with SFTTrainer.
    
    Returns LoraConfig that can be passed directly to SFTTrainer's peft_config parameter.
    SFTTrainer will automatically apply the LoRA adapters during training.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


# Keep apply_lora for backward compatibility (used by eval/inference code)
def apply_lora(
    model,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
):
    """Apply LoRA to a model (for backward compatibility with eval/inference code).
    
    Note: For training, use create_lora_config() and pass to SFTTrainer instead.
    """
    from peft import get_peft_model
    
    cfg = create_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, cfg)
    return model
