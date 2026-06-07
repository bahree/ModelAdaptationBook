"""Verifiable reward function for an RFT-style task (chapter 8, section 8.5).

A reward function for reinforcement fine-tuning (RFT) takes a single model
response and returns a scalar score computed by a deterministic checker, not by
a human or a learned reward model. This example scores a structured-extraction
task: the model is asked to extract invoice fields as JSON, and the reward is
the fraction of required fields that parse and match the ground truth, with a
hard zero when the output is not valid JSON. The check is deterministic and
fast enough to run inside the RFT training loop.

Run the demo:
    python -m chapter08.scripts.verifiable_reward
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

REQUIRED_FIELDS = ("invoice_number", "amount", "due_date")


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the first JSON object out of a model response, which may wrap it in
    prose or a ```json fence. Returns None if nothing parses."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        span = re.search(r"\{.*\}", text, re.S)   # first {...} span
        candidate = span.group(0) if span else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def invoice_reward(response: str, expected: dict[str, Any]) -> float:
    """Verifiable reward in [0.0, 1.0] for an invoice-extraction response.

    Deterministic checker (no human, no learned reward model):
      * 0.0 if the response is not valid JSON (the verifier rejects it);
      * otherwise the fraction of REQUIRED_FIELDS whose value matches expected.
    """
    parsed = extract_json(response)
    if parsed is None:
        return 0.0
    hits = sum(
        1 for field in REQUIRED_FIELDS
        if str(parsed.get(field, "")).strip() == str(expected.get(field, "")).strip()
    )
    return hits / len(REQUIRED_FIELDS)


def main() -> None:
    expected = {"invoice_number": "INV-2026-0413",
                "amount": "1240.00", "due_date": "2026-07-01"}
    samples = {
        "correct (fenced JSON)":
            '```json\n{"invoice_number": "INV-2026-0413", '
            '"amount": "1240.00", "due_date": "2026-07-01"}\n```',
        "partial (1 field wrong)":
            '{"invoice_number": "INV-2026-0413", "amount": "1240.00", '
            '"due_date": "2026-09-30"}',
        "not JSON (prose)":
            "The invoice number is INV-2026-0413 and the amount is 1240.00.",
    }
    for label, response in samples.items():
        print(f"{label:24} reward = {invoice_reward(response, expected):.2f}")


if __name__ == "__main__":
    main()
