"""Core evaluation functions for Chapter 5 fine-tuning experiments.

Provides evaluation on three axes:
    1. **Instruction-following** - Dolly test set with exact match and token F1.
    2. **Safety** - A suite of 10 harmful prompts to measure refusal rate regression.
    3. **Toy golden set** - Simple Q&A pairs to sanity-check model behavior.

Also includes loss/perplexity computation on held-out JSONL data, and report
generation (JSON + Markdown). Used by ``scripts/listing_5_3_evaluate.py``
(Listing 5.3) to compare base model vs. adapter variants.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chat_template import build_prompt_text
from .metrics import exact_match, is_refusal, token_f1
from common.jsonl import read_jsonl_list


def generate_completion(
    model,
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Generate a single completion for a user prompt using greedy decoding.

    Builds a chat-formatted prompt (system + user), runs model.generate with
    do_sample=False for deterministic (reproducible) evaluation, and returns
    only the newly generated tokens.

    Args:
        model: A HuggingFace causal LM (base or with adapter attached).
        tokenizer: Matching tokenizer.
        system_prompt: System message prepended to the conversation.
        user_prompt: The user's question or instruction.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's response as a decoded string (special tokens removed).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = build_prompt_text(tokenizer, messages)
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # do_sample=False for deterministic output -- essential for reproducible
        # evaluation scores across runs.
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Slice off the prompt tokens so we only decode the model's response.
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def compute_token_level_loss(model, batch) -> Tuple[float, int]:
    """Return (sum_loss, token_count) for this batch, using labels and ignoring -100."""
    import torch.nn.functional as F

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    logits = outputs.logits
    labels = batch["labels"]
    # shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    ).item()
    token_count = int((shift_labels != -100).sum().item())
    return loss_sum, token_count


def eval_loss_on_jsonl(
    model,
    tokenizer,
    *,
    jsonl_path: str | Path,
    system_prompt: str,
    max_length: int,
    max_examples: int = 200,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Compute average cross-entropy loss and perplexity on a held-out JSONL dataset.

    Encodes chat examples, runs a forward pass (no generation), and computes
    token-level loss only on assistant (completion) tokens. Useful for comparing
    adapters without the cost of full generation.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        jsonl_path: Path to chat-format JSONL file.
        system_prompt: System message to prepend.
        max_length: Maximum sequence length for tokenization.
        max_examples: Cap on number of examples to evaluate.
        batch_size: Batch size for the dataloader.

    Returns:
        Dict with ``loss`` (avg cross-entropy), ``perplexity`` (exp of loss),
        and ``tokens`` (total tokens evaluated).
    """
    from .data import load_chat_jsonl
    from .dataset import encode_examples


    examples = load_chat_jsonl(jsonl_path, system_prompt=system_prompt)[:max_examples]
    ds = encode_examples(tokenizer, examples, max_length=max_length)

    from torch.utils.data import DataLoader
    import torch

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad input_ids and labels to the same length within a batch.

        Labels are padded with -100, the HuggingFace convention for "ignore
        this token in the loss computation" (used to mask prompt tokens so
        only assistant tokens contribute to the loss).
        """
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f.get("attention_mask") for f in features]
        labels = [f["labels"] for f in features]
        batch = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            lab = list(lab)
            if len(lab) < max_len:
                # -100 = ignore index in CrossEntropyLoss (HF convention)
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    for batch in dl:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss_sum, token_count = compute_token_level_loss(model, batch)
        total_loss += loss_sum
        total_tokens += token_count

    avg_loss = total_loss / max(1, total_tokens)
    # Perplexity = exp(avg_loss). Capped at loss < 50 to avoid overflow.
    ppl = float(math.exp(avg_loss)) if avg_loss < 50 else float("inf")
    return {"loss": float(avg_loss), "perplexity": ppl, "tokens": float(total_tokens)}


def eval_dolly_test_set(
    model,
    tokenizer,
    *,
    test_jsonl: str | Path,
    system_prompt: str,
    max_new_tokens: int = 256,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model on Dolly test set with per-category metrics.
    
    The test JSONL should have messages format. Category information is stored
    in a separate field or can be loaded from the manifest.
    
    Returns metrics including:
    - Overall exact match and token F1
    - Per-category exact match and token F1 (if category info available)
    - Category distribution
    """
    from .data import load_chat_jsonl
    
    examples = load_chat_jsonl(test_jsonl, system_prompt=system_prompt)
    if max_examples:
        examples = examples[:max_examples]
    
    # Try to load category info from original rows or manifest
    rows = read_jsonl_list(test_jsonl)
    manifest_path = Path(test_jsonl).parent / "manifest.json"
    category_dist = None
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        category_dist = manifest.get("category_distribution", {})
    
    items = []
    ems = 0
    f1s: List[float] = []
    category_ems: Dict[str, List[bool]] = {}
    category_f1s: Dict[str, List[float]] = {}
    
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating examples...", total=len(examples))
        
        for idx, (example, row) in enumerate(zip(examples, rows)):
            # Get user prompt and reference response
            messages = example.messages
            user_prompt = None
            reference = None
            
            for msg in messages:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "assistant":
                    reference = msg["content"]
            
            if not user_prompt or not reference:
                progress.update(task, advance=1)
                continue
            
            # Get category if available
            category = row.get("category", "unknown")
            # If not in row, try to infer from category_dist if we have it
            if category == "unknown" and category_dist and len(category_dist) > 0:
                # We can't reliably map, so use "unknown"
                pass
            
            if category not in category_ems:
                category_ems[category] = []
                category_f1s[category] = []
            
            # Generate prediction
            pred = generate_completion(
                model,
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
            
            # Compute metrics
            em = exact_match(pred, reference)
            f1 = token_f1(pred, reference)
            
            ems += 1 if em else 0
            f1s.append(f1)
            category_ems[category].append(em)
            category_f1s[category].append(f1)
            
            items.append({
                "id": idx,
                "category": category,
                "pred": pred,
                "ref": reference,
                "em": em,
                "f1": f1,
            })
            
            progress.update(task, advance=1)
    
    # Compute per-category metrics
    category_metrics = {}
    for cat in category_ems:
        cat_ems = category_ems[cat]
        cat_f1s = category_f1s[cat]
        category_metrics[cat] = {
            "count": len(cat_ems),
            "exact_match": sum(cat_ems) / max(1, len(cat_ems)),
            "token_f1": sum(cat_f1s) / max(1, len(cat_f1s)),
        }
    
    return {
        "count": len(items),
        "exact_match": ems / max(1, len(items)),
        "token_f1": float(sum(f1s) / max(1, len(f1s))),
        "category_metrics": category_metrics,
        "items": items,
    }


def safety_suite(
    model,
    tokenizer,
    *,
    prompts_jsonl: str | Path,
    system_prompt: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Run a safety evaluation: measure how often the model refuses harmful prompts.

    The safety prompts JSONL contains 10 adversarial/harmful prompts across
    categories (violence, illegal activity, etc.). A well-aligned model should
    refuse all of them (100% refusal rate). A drop in refusal rate after
    fine-tuning indicates safety regression -- a critical issue to monitor.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompts_jsonl: Path to safety prompts JSONL file.
        system_prompt: System message to use.
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        Dict with ``count``, ``refusal_rate`` (float 0-1), and ``items``
        (list of per-prompt results).
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    rows = read_jsonl_list(prompts_jsonl)
    results = []
    refusals = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Running safety checks...", total=len(rows))
        
        for row in rows:
            prompt = row.get("prompt", "")
            completion = generate_completion(
                model,
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            refused = is_refusal(completion)
            refusals += 1 if refused else 0
            results.append(
                {
                    "id": row.get("id"),
                    "category": row.get("category"),
                    "prompt": prompt,
                    "completion": completion,
                    "refusal": refused,
                }
            )
            progress.update(task, advance=1)
    
    return {
        "count": len(results),
        "refusal_rate": (refusals / max(1, len(results))),
        "items": results,
    }


def load_model_variant(
    *,
    base_model: str,
    adapter: str | None,
):
    """Load the base model and optionally attach a LoRA adapter for evaluation.

    The model is reloaded from scratch for each variant to ensure a clean state
    (no leftover adapter weights from a previous variant).

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        adapter: Path to a LoRA adapter directory, or None for base-only evaluation.

    Returns:
        Tuple of (model, tokenizer).
    """
    from peft import PeftModel
    from common.hub import split_ref, subfolder_kwargs
    from .modeling import load_base_model_lora, load_tokenizer

    tokenizer = load_tokenizer(base_model)
    # gradient_checkpointing=False because we only run forward passes for eval.
    model = load_base_model_lora(base_model, gradient_checkpointing=False)
    if adapter:
        ap_path, ap_sub = split_ref(adapter)
        model = PeftModel.from_pretrained(model, ap_path, **subfolder_kwargs(ap_sub))
    return model, tokenizer


def eval_toy_golden(
    model,
    tokenizer,
    *,
    golden_jsonl: str | Path,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Evaluate model on a small "toy" golden set of simple Q&A pairs.

    The golden set is a handful of straightforward questions with known
    answers (e.g., "What is the capital of France?"). It serves as a
    quick sanity check that the model generates coherent responses.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        golden_jsonl: Path to toy golden JSONL file (messages + reference).
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        Dict with ``count``, ``exact_match``, ``token_f1``, and ``items``.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    rows = read_jsonl_list(golden_jsonl)
    items = []
    ems = 0
    f1s: List[float] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating toy test set...", total=len(rows))
        
        for row in rows:
            msgs = row["messages"]
            ref = row["reference"]
            user_prompt = msgs[-1]["content"]
            system_prompt = msgs[0]["content"]
            pred = generate_completion(
                model,
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)
            ems += 1 if em else 0
            f1s.append(f1)
            items.append({"id": row.get("id"), "pred": pred, "ref": ref, "em": em, "f1": f1})
            progress.update(task, advance=1)

    return {
        "count": len(items),
        "exact_match": ems / max(1, len(items)),
        "token_f1": float(sum(f1s) / max(1, len(f1s))),
        "items": items,
    }


def write_report(path: str | Path, obj: Dict[str, Any]) -> None:
    """Write an evaluation results dict as a JSON file.

    The JSON report is the machine-readable counterpart to the human-readable
    Markdown summary generated by ``listing_5_3_evaluate.py``. Both are saved
    to the same output directory (e.g., ``chapter05/runs/eval_report/``).

    Args:
        path: Destination file path (e.g., ``report.json``).
        obj: Dict of evaluation results to serialize.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
