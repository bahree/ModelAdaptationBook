"""Capture a side-by-side comparison: Ch6 SFT model vs frontier APIs.

Used to ground the Chapter 7 opening hook in a concrete failure of the
local Ch6 SFT model on the same VPN-troubleshooting prompt that recurs
through Chapters 5, 6, and 8. The frontier responses come from
OpenRouter (one key, multiple vendors).

Outputs a single JSON to ``chapter07/runs/frontier_comparison.json``
with:

- the prompt and the system instruction used,
- the Ch6 SFT model's response (loaded locally with ``device_map=auto``),
- responses from each frontier model, with usage and cost,
- ISO timestamp + each model's exact id-as-served (for stable footnoting).

Default frontier set is Claude Sonnet 4.5, Gemini 2.5 Pro, DeepSeek
V3.1. Override with ``--frontier_models`` if needed.

Run from code/ (with the venv activated and OPENROUTER_API_KEY in
``code/.env``):

    python -m chapter07.scripts.capture_frontier_comparison \
        --prompt "How do I troubleshoot a VPN connection failure?" \
        --sft_dir chapter06/runs/sft_run1 \
        --output chapter07/runs/frontier_comparison.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Dict, List

from common.openrouter import OpenRouterError, chat

DEFAULT_PROMPT = "How do I troubleshoot a VPN connection failure?"
DEFAULT_SYSTEM = "You are an IT support assistant."
DEFAULT_FRONTIER_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat-v3.1",
]


def run_local_sft(
    sft_dir: str, prompt: str, system: str, max_new_tokens: int
) -> Dict[str, object]:
    """Run the prompt through the local Ch6 SFT model.

    Imported lazily so the script does not pay the transformers cost when
    only the frontier comparison is needed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(sft_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        sft_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    content = tok.decode(gen_ids, skip_special_tokens=True).strip()
    elapsed = time.time() - started

    return {
        "label": "Ch6 SFT (Qwen3-4B-Instruct-2507)",
        "model": sft_dir,
        "content": content,
        "wall_seconds": round(elapsed, 2),
        "tokens_generated": int(gen_ids.shape[0]),
    }


def run_openrouter_model(prompt: str, system: str, model: str, max_tokens: int) -> Dict[str, object]:
    """Call one OpenRouter model and return a uniform record."""
    started = time.time()
    result = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - started
    usage = result["usage"]
    return {
        "label": model,
        "model": result["model"],
        "content": result["content"],
        "reasoning": result.get("reasoning"),
        "wall_seconds": round(elapsed, 2),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": usage.get("cost"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture SFT vs frontier-API comparison")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--sft_dir", default="chapter06/runs/sft_run1")
    parser.add_argument(
        "--frontier_models",
        nargs="*",
        default=DEFAULT_FRONTIER_MODELS,
        help="OpenRouter model ids to query",
    )
    parser.add_argument("--max_tokens_local", type=int, default=400)
    parser.add_argument("--max_tokens_remote", type=int, default=1024)
    parser.add_argument(
        "--skip_local",
        action="store_true",
        help="Only call the frontier models (useful when the SFT model is unavailable)",
    )
    parser.add_argument(
        "--output",
        default="chapter07/runs/frontier_comparison.json",
    )
    args = parser.parse_args()

    print(f"Prompt: {args.prompt}")
    print(f"System: {args.system}")

    responses: List[Dict[str, object]] = []

    if not args.skip_local:
        print(f"\n--- Local: Ch6 SFT model from {args.sft_dir} ---")
        local = run_local_sft(args.sft_dir, args.prompt, args.system, args.max_tokens_local)
        responses.append(local)
        print(f"  done in {local['wall_seconds']}s, {local['tokens_generated']} tokens")

    total_cost = 0.0
    for model_id in args.frontier_models:
        print(f"\n--- Frontier: {model_id} (via OpenRouter) ---")
        try:
            rec = run_openrouter_model(args.prompt, args.system, model_id, args.max_tokens_remote)
            responses.append(rec)
            cost = rec["usage"].get("cost_usd") or 0.0
            total_cost += cost
            print(
                f"  done in {rec['wall_seconds']}s, "
                f"{rec['usage'].get('completion_tokens', '?')} completion tokens, "
                f"${cost:.5f}"
            )
        except OpenRouterError as exc:
            print(f"  FAILED: {exc}")
            responses.append({"label": model_id, "model": model_id, "error": str(exc)})

    payload = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "prompt": args.prompt,
        "system": args.system,
        "responses": responses,
        "total_openrouter_cost_usd": round(total_cost, 5),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("=" * 60)
    print("Comparison saved to:")
    print(f"  {out_path}")
    print(f"Total OpenRouter spend on this run: ${total_cost:.5f}")
    print()
    print("First ~200 chars of each response (use the JSON for full text):")
    for rec in responses:
        if "error" in rec:
            print(f"  [{rec['label']}] ERROR: {rec['error'][:140]}")
            continue
        snippet = rec["content"].replace("\n", " ")[:200]
        print(f"  [{rec['label']}] {snippet}{'...' if len(rec['content']) > 200 else ''}")


if __name__ == "__main__":
    main()
