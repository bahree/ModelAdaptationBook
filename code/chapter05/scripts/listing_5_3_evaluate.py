"""Evaluation script comparing base model vs. fine-tuned adapter(s) (Listing 5.3).

Loads the base model, evaluates it, then loads one or two adapters and evaluates
each, computing per-metric deltas. Produces both a machine-readable JSON report
and a human-readable Markdown summary.

Usage (base vs. single adapter):
    python chapter05/scripts/listing_5_3_evaluate.py \\
        --base Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter chapter05/runs/it_lora \\
        --dolly_test data/it_support/valid.jsonl

Usage (base vs. LoRA vs. QLoRA):
    python chapter05/scripts/listing_5_3_evaluate.py \\
        --base Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter chapter05/runs/it_lora \\
        --adapter_alt chapter05/runs/it_qlora \\
        --dolly_test data/it_support/valid.jsonl

See Chapter 5, Section 5.1 (Step 3) for walkthrough and expected results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from chapter05.eval import (
    eval_dolly_test_set,
    eval_toy_golden,
    load_model_variant,
    safety_suite,
    write_report,
)
from chapter05.chat_template import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Returns:
        Namespace with base model, adapter paths, test data paths,
        generation settings, and output directory.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model: accepts a local path, an HF repo id, or 'repo#subfolder' "
        "(e.g. bahree/ModelAdaptationBook#ch6-sft)",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        help="Adapter for main run: accepts a local path, an HF repo id, or "
        "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch5-lora)",
    )
    ap.add_argument(
        "--adapter_alt",
        default=None,
        help="Adapter for comparison run: accepts a local path, an HF repo id, or "
        "'repo#subfolder' (e.g. bahree/ModelAdaptationBook#ch8-dpo-lora)",
    )

    ap.add_argument(
        "--dolly_test",
        default=None,
        help="Test set JSONL path for primary evaluation (e.g. data/it_support/valid.jsonl)",
    )
    ap.add_argument("--toy_golden", default="chapter05/data/golden/toy_test.jsonl", help="Toy test set (optional)")
    ap.add_argument("--safety_prompts", default="chapter05/data/golden/safety_regression_prompts.jsonl")

    ap.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for safety suite (toy golden uses per-example system prompt).",
    )

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--out", default="chapter05/runs/eval_report", help="Output folder for reports")
    return ap.parse_args()


def summarize_variant(name: str, model, tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full evaluation suite for a single model variant.

    Evaluates on the Dolly test set (instruction-following), the toy golden
    set (sanity check), and the safety suite (refusal rate). Returns a dict
    of all results for this variant.

    Args:
        name: Label for this variant (e.g., "base", "adapter", "adapter_alt").
        model: A HuggingFace causal LM (base or with adapter attached).
        tokenizer: Matching tokenizer.
        args: Parsed CLI arguments with test data paths and generation settings.

    Returns:
        Dict with evaluation results keyed by test type ("dolly", "toy", "safety").
    """
    result: Dict[str, Any] = {"name": name}
    
    # Primary evaluation: Dolly test set
    if args.dolly_test and Path(args.dolly_test).exists():
        dolly_result = eval_dolly_test_set(
            model,
            tokenizer,
            test_jsonl=args.dolly_test,
            system_prompt=args.system_prompt,
            max_new_tokens=256,
        )
        result["dolly"] = dolly_result
    
    # Legacy evaluations (optional)
    if Path(args.toy_golden).exists():
        result["toy"] = eval_toy_golden(
            model, tokenizer, golden_jsonl=args.toy_golden, max_new_tokens=args.max_new_tokens
        )
    
    result["safety"] = safety_suite(
        model,
        tokenizer,
        prompts_jsonl=args.safety_prompts,
        system_prompt=args.system_prompt,
    )
    
    return result


def main() -> None:
    """Evaluate base model and adapter(s), compute deltas, and write reports."""
    from rich.console import Console
    console = Console()
    
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Step 1/4:[/bold cyan] Loading base model...")
    base_model, base_tok = load_model_variant(base_model=args.base, adapter=None)
    console.print("[green]✓[/green] Base model loaded\n")
    
    console.print("[bold cyan]Step 2/4:[/bold cyan] Evaluating base model...")
    base_res = summarize_variant("base", base_model, base_tok, args)
    console.print("[green]✓[/green] Base evaluation complete\n")

    # Free each model before loading the next so the base + adapter (+ optional
    # alternative adapter) never sit in GPU memory at the same time. Without this,
    # a single 24 GB GPU runs out of memory on the third model and offloads it to
    # CPU, which makes the comparison evaluation unusably slow.
    import gc
    import torch as _torch
    del base_model, base_tok
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    res: Dict[str, Any] = {"base": base_res}

    if args.adapter:
        console.print(f"[bold cyan]Step 3/4:[/bold cyan] Loading adapter from {args.adapter}...")
        m, t = load_model_variant(base_model=args.base, adapter=args.adapter)
        console.print("[green]✓[/green] Adapter loaded\n")
        
        console.print("[bold cyan]Step 4/4:[/bold cyan] Evaluating fine-tuned model...")
        res["adapter"] = summarize_variant("adapter", m, t, args)
        console.print("[green]✓[/green] Fine-tuned evaluation complete\n")
        del m, t
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    if args.adapter_alt:
        console.print(f"[bold cyan]Loading alternative adapter from {args.adapter_alt}...[/bold cyan]")
        m, t = load_model_variant(base_model=args.base, adapter=args.adapter_alt)
        console.print("[green]✓[/green] Alternative adapter loaded\n")
        
        console.print("[bold cyan]Evaluating alternative adapter...[/bold cyan]")
        res["adapter_alt"] = summarize_variant("adapter_alt", m, t, args)
        console.print("[green]✓[/green] Alternative evaluation complete\n")

    def maybe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """Compute a - b, returning None if either value is missing."""
        if a is None or b is None:
            return None
        return float(a - b)

    def compute_deltas(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metric deltas between an adapter and the base model."""
        deltas: Dict[str, Any] = {
            "safety": {
                "refusal_rate": maybe_delta(other["safety"]["refusal_rate"], base["safety"]["refusal_rate"]),
            },
        }
        
        # Dolly metrics (primary)
        if base.get("dolly") and other.get("dolly"):
            deltas["dolly"] = {
                "exact_match": maybe_delta(other["dolly"]["exact_match"], base["dolly"]["exact_match"]),
                "token_f1": maybe_delta(other["dolly"]["token_f1"], base["dolly"]["token_f1"]),
                "category_metrics": {},
            }
            # Per-category deltas
            base_cats = base["dolly"].get("category_metrics", {})
            other_cats = other["dolly"].get("category_metrics", {})
            for cat in set(base_cats.keys()) | set(other_cats.keys()):
                if cat in base_cats and cat in other_cats:
                    deltas["dolly"]["category_metrics"][cat] = {
                        "exact_match": maybe_delta(
                            other_cats[cat]["exact_match"], base_cats[cat]["exact_match"]
                        ),
                        "token_f1": maybe_delta(
                            other_cats[cat]["token_f1"], base_cats[cat]["token_f1"]
                        ),
                    }
        
        # Toy metrics (optional)
        if base.get("toy") and other.get("toy"):
            deltas["toy"] = {
                "exact_match": maybe_delta(other["toy"]["exact_match"], base["toy"]["exact_match"]),
                "token_f1": maybe_delta(other["toy"]["token_f1"], base["toy"]["token_f1"]),
            }
        
        return deltas

    if "adapter" in res:
        res["adapter_deltas_vs_base"] = compute_deltas(res["base"], res["adapter"])
    if "adapter_alt" in res:
        res["adapter_alt_deltas_vs_base"] = compute_deltas(res["base"], res["adapter_alt"])

    # Write JSON
    write_report(out_dir / "report.json", res)

    # Write a short Markdown summary
    def fmt_pct(x: float) -> str:
        """Format a 0-1 float as a percentage string (e.g., 0.6 -> '60.0%')."""
        return f"{x*100:.1f}%"

    def fmt_delta(x: Optional[float], *, pct: bool = False) -> str:
        """Format a delta value with +/- sign (e.g., +0.1321 or +13.2%)."""
        if x is None:
            return "n/a"
        if pct:
            return f"{x*100:+.1f}%"
        return f"{x:+.4f}"

    lines = []
    lines.append("# Chapter 5 Evaluation Report")
    lines.append("")
    lines.append(f"- Base model: `{args.base}`")
    lines.append(f"- System prompt: `{args.system_prompt}`")
    if args.dolly_test:
        lines.append(f"- Dolly test set: `{args.dolly_test}`")
    if args.adapter:
        lines.append(f"- Adapter: `{args.adapter}`")
    if args.adapter_alt:
        lines.append(f"- Adapter (alt): `{args.adapter_alt}`")
    lines.append("")

    for key in ["base", "adapter", "adapter_alt"]:
        if key not in res:
            continue
        variant = res[key]
        lines.append(f"## {key}")
        
        # Primary: Dolly metrics
        if variant.get("dolly"):
            d = variant["dolly"]
            lines.append("### Dolly Test Set (Instruction-Following)")
            lines.append(f"- **Overall exact match**: {fmt_pct(d['exact_match'])}")
            lines.append(f"- **Overall token-F1**: {d['token_f1']:.3f}")
            lines.append(f"- **Test examples**: {d['count']}")
            if d.get("category_metrics"):
                lines.append("\n**Per-Category Accuracy:**")
                for cat, metrics in sorted(d["category_metrics"].items()):
                    lines.append(
                        f"- {cat}: EM={fmt_pct(metrics['exact_match'])}, F1={metrics['token_f1']:.3f} "
                        f"(n={metrics['count']})"
                    )
            lines.append("")
        
        # Safety
        safety = variant["safety"]
        lines.append(f"- **Safety refusal rate**: {fmt_pct(safety['refusal_rate'])}")
        
        # Toy metrics (if present)
        if variant.get("toy"):
            toy = variant["toy"]
            lines.append(f"- **Toy exact match**: {fmt_pct(toy['exact_match'])}")
            lines.append(f"- **Toy token-F1**: {toy['token_f1']:.3f}")
        
        lines.append("")

    # Delta section (base vs adapters)
    def add_delta_block(delta_key: str, label: str) -> None:
        """Append a Markdown section showing metric deltas vs. base."""
        if delta_key not in res:
            return
        d = res[delta_key]
        lines.append(f"## {label} (Improvement vs Base)")
        
        # Primary: Dolly metrics
        if d.get("dolly"):
            lines.append("### Dolly Test Set Improvements")
            lines.append(f"- **Overall exact match Δ**: {fmt_delta(d['dolly']['exact_match'], pct=True)}")
            lines.append(f"- **Overall token-F1 Δ**: {fmt_delta(d['dolly']['token_f1'])}")
            if d["dolly"].get("category_metrics"):
                lines.append("\n**Per-Category Improvements:**")
                for cat, metrics in sorted(d["dolly"]["category_metrics"].items()):
                    em_delta = metrics.get("exact_match")
                    f1_delta = metrics.get("token_f1")
                    if em_delta is not None or f1_delta is not None:
                        em_str = fmt_delta(em_delta, pct=True) if em_delta is not None else "n/a"
                        f1_str = fmt_delta(f1_delta) if f1_delta is not None else "n/a"
                        lines.append(f"- {cat}: EM Δ={em_str}, F1 Δ={f1_str}")
            lines.append("")
        
        # Safety
        lines.append(f"- **Safety refusal rate Δ**: {fmt_delta(d['safety']['refusal_rate'], pct=True)}")
        
        # Toy metrics
        if d.get("toy"):
            lines.append(f"- **Toy exact match Δ**: {fmt_delta(d['toy']['exact_match'], pct=True)}")
            lines.append(f"- **Toy token-F1 Δ**: {fmt_delta(d['toy']['token_f1'])}")
        
        lines.append("")

    add_delta_block("adapter_deltas_vs_base", "adapter")
    add_delta_block("adapter_alt_deltas_vs_base", "adapter_alt")

    console.print("\n[bold cyan]Writing evaluation reports...[/bold cyan]")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    console.print("\n[bold green]✓ Evaluation complete![/bold green]")
    console.print(f"[green]✓[/green] JSON report: {out_dir / 'report.json'}")
    console.print(f"[green]✓[/green] Markdown summary: {out_dir / 'report.md'}")
    console.print("\n[yellow]→[/yellow] View the markdown report for a human-readable summary")


if __name__ == "__main__":
    main()
