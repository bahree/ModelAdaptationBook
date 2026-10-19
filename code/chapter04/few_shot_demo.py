"""Listing 4.1 -- Few-shot prompt template for ticket classification.

Build a few-shot prompt from a labelled example bank, run it against a
held-out test set, and report top-1 accuracy.  The script supports two
backends:

- ``hf`` (default): loads ``Qwen/Qwen3-4B-Instruct-2507`` via transformers.
  Requires a GPU (~12 GB VRAM) and the standard transformers stack.
- ``mock``: a deterministic offline backend that uses keyword matching.
  Useful for smoke tests on machines without a GPU and for unit tests.

Run from code/:

    # GPU (Qwen3-4B):
    python -m chapter04.few_shot_demo \
        --bank chapter04/data/example_bank.jsonl \
        --test chapter04/data/held_out_test.jsonl \
        --shots 8 --output chapter04/runs/few_shot_report.json

    # CPU (offline mock backend, smoke test only):
    python -m chapter04.few_shot_demo \
        --bank chapter04/data/example_bank.jsonl \
        --test chapter04/data/held_out_test.jsonl \
        --shots 8 --backend mock \
        --output chapter04/runs/few_shot_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from common.device import resolve_device_map
from common.jsonl import read_jsonl_list, write_jsonl
from common.seed import seed_everything

from chapter04 import CATEGORIES, DEFAULT_MODEL_NAME, DEFAULT_SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# Prompt assembly (Listing 4.1)
# ---------------------------------------------------------------------------


def format_example(ticket: str, category: str) -> str:
    """Render a single labelled example in the canonical schema."""
    return f"Ticket: {ticket}\nCategory: {category}"


def build_prompt(examples: Sequence[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    """Assemble a chat-format few-shot prompt.

    Args:
        examples: Sequence of dicts with ``ticket`` and ``category`` keys,
            ordered by relevance with the most-relevant last (the recency
            rule from section 4.2).
        query: The ticket text to classify.

    Returns:
        A list of chat messages compatible with ``apply_chat_template``.
    """
    shots = "\n\n".join(format_example(e["ticket"], e["category"]) for e in examples)
    user_content = f"{shots}\n\nTicket: {query}\nCategory:"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------


def stratified_sample(
    bank: List[Dict[str, str]], shots: int, *, rng: random.Random
) -> List[Dict[str, str]]:
    """Pick ``shots`` examples spread across categories.

    Falls back to random sampling when the bank does not have enough
    categories to fill the budget.
    """
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for ex in bank:
        by_category.setdefault(ex["category"], []).append(ex)

    selected: List[Dict[str, str]] = []
    cats = list(by_category)
    rng.shuffle(cats)
    while len(selected) < shots and cats:
        for cat in list(cats):
            if not by_category[cat]:
                cats.remove(cat)
                continue
            selected.append(by_category[cat].pop(rng.randrange(len(by_category[cat]))))
            if len(selected) >= shots:
                break
    return selected


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------


def make_hf_backend(
    model_name: str, *, temperature: float = 0.0, top_p: float = 0.95
) -> Callable[[List[Dict[str, str]]], str]:
    """Return a callable that takes a chat-format prompt and returns text.

    When ``temperature`` is 0, decoding is greedy (deterministic).  When it
    is positive, sampling is enabled with ``top_p`` nucleus filtering, which
    is what the prompt validator uses to surface run-to-run variability.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=resolve_device_map(), trust_remote_code=True
    )
    model.eval()

    sample = temperature > 0.0
    gen_kwargs = {"max_new_tokens": 8, "do_sample": sample}
    if sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    def generate(messages: List[Dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return generate


def make_mock_backend() -> Callable[[List[Dict[str, str]]], str]:
    """Deterministic offline backend driven by keyword matching.

    Used for unit tests and CPU-only smoke tests.  The backend looks at the
    last ``Ticket:`` block in the user message and applies a small set of
    keyword rules.  Accuracy is meaningfully below a real model, which is
    fine; the goal is to exercise the prompt-assembly path end to end.
    """
    rules = [
        ("billing", ["refund", "charged", "invoice", "subscription", "card", "billing"]),
        ("login", ["password", "log in", "login", "sso", "single sign", "2fa", "two-factor"]),
        ("integrations", ["salesforce", "slack", "hubspot", "webhook", "integration", "okta"]),
        ("mobile", ["ios", "android", "app crash", "mobile app", "push notification"]),
        ("performance", ["slow", "timeout", "504", "load time", "doubled"]),
        ("data_export", ["export", "csv", "download", "bulk export"]),
        ("api", ["rest api", "rate limit", "api key", "429", "endpoint"]),
        ("security", ["security", "soc 2", "saml", "compliance", "encryption", "audit"]),
        ("onboarding", ["sign up", "signed up", "invite", "workspace", "tutorial", "trial"]),
        ("feature_request", ["please add", "could you add", "would love", "feature request", "support for"]),
        ("bug_report", ["does nothing", "bug", "crash", "error", "wrong", "broken", "freezes"]),
    ]

    def generate(messages: List[Dict[str, str]]) -> str:
        user_text = messages[-1]["content"].lower()
        last_query = user_text.rsplit("ticket:", 1)[-1]
        for category, keywords in rules:
            if any(kw in last_query for kw in keywords):
                return category
        return "other"

    return generate


BACKENDS: Dict[str, Callable[[argparse.Namespace], Callable[[List[Dict[str, str]]], str]]] = {
    "hf": lambda args: make_hf_backend(
        args.model, temperature=getattr(args, "temperature", 0.0)
    ),
    "mock": lambda args: make_mock_backend(),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def parse_category(raw: str) -> str:
    """Extract a category label from a model response."""
    raw = raw.strip().lower().split("\n")[0].strip(" .,'\"`")
    for cat in CATEGORIES:
        if raw.startswith(cat):
            return cat
    return raw


def evaluate(
    examples: Sequence[Dict[str, str]],
    test_set: Sequence[Dict[str, str]],
    backend: Callable[[List[Dict[str, str]]], str],
) -> Dict[str, object]:
    """Run the few-shot prompt against the test set and return a report."""
    correct = 0
    rows: List[Dict[str, object]] = []
    for case in test_set:
        messages = build_prompt(examples, case["ticket"])
        raw = backend(messages)
        prediction = parse_category(raw)
        is_correct = prediction == case["category"]
        if is_correct:
            correct += 1
        rows.append(
            {
                "ticket": case["ticket"],
                "expected": case["category"],
                "predicted": prediction,
                "raw": raw,
                "correct": is_correct,
            }
        )
    accuracy = correct / max(len(test_set), 1)
    return {
        "shots": len(examples),
        "test_size": len(test_set),
        "top1_accuracy": round(accuracy, 4),
        "correct": correct,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot ticket classifier (chapter 4)")
    parser.add_argument(
        "--bank",
        default="chapter04/data/example_bank.jsonl",
        help="JSONL file with `ticket` and `category` fields",
    )
    parser.add_argument(
        "--test",
        default="chapter04/data/held_out_test.jsonl",
        help="JSONL held-out test set",
    )
    parser.add_argument("--shots", type=int, default=8, help="Number of few-shot examples")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="hf",
        help="Model backend (hf = transformers, mock = offline keyword classifier)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model id (hf backend)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the hf backend (0 = greedy decoding)",
    )
    parser.add_argument(
        "--output",
        default="chapter04/runs/few_shot_report.json",
        help="Where to write the report JSON",
    )
    parser.add_argument(
        "--details",
        default="chapter04/runs/few_shot_details.jsonl",
        help="Where to write per-example predictions",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)

    bank = read_jsonl_list(args.bank)
    test_set = read_jsonl_list(args.test)
    if args.shots > len(bank):
        raise ValueError(
            f"Requested {args.shots} shots but the example bank only has {len(bank)} entries."
        )

    examples = stratified_sample(bank, args.shots, rng=rng)
    backend = BACKENDS[args.backend](args)

    print(f"Running few-shot ({args.shots} shots) on {len(test_set)} test cases")
    print(f"Backend: {args.backend}")

    started = time.time()
    report = evaluate(examples, test_set, backend)
    report["wall_seconds"] = round(time.time() - started, 2)
    report["backend"] = args.backend
    report["model"] = args.model if args.backend == "hf" else None
    report["temperature"] = args.temperature if args.backend == "hf" else None
    report["selected_examples"] = [
        {"ticket": e["ticket"], "category": e["category"]} for e in examples
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.pop("rows")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(args.details, rows)

    print(f"\nTop-1 accuracy: {report['top1_accuracy']:.2%} ({report['correct']}/{report['test_size']})")
    print(f"Report:  {out_path}")
    print(f"Details: {args.details}")


if __name__ == "__main__":
    main()
