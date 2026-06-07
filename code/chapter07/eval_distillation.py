"""Evaluate distillation quality: compare base vs teacher vs student.

Three-way comparison using the same metrics as Chapters 5 and 6.

Reference selection (--reference):
    human   (default) score every model against the ORIGINAL human Dolly
            answer for each prompt. This is the methodologically correct
            choice for measuring distillation retention: the reference is
            independent of all three models, so you get the expected
            base < student < teacher ordering and a meaningful retention
            percentage.
    teacher score against the teacher's own generated output (the old
            behavior). This measures fidelity-to-teacher, not quality.
            Because the student is TRAINED to reproduce those teacher
            outputs and the teacher is re-generated with greedy decoding
            (while the references were sampled at temperature 0.7), this
            mode makes the student look as good as or better than the
            teacher (student ~= 100%+), which is an artifact, not a gain.

Run from code/:
    python -m chapter07.eval_distillation \
        --data_dir chapter07/data/distill_ready \
        --teacher_dir chapter06/runs/sft_run1 \
        --student_dir chapter07/runs/student_run1 \
        --dolly_dir chapter06/data/dolly_sft \
        --reference human \
        --output chapter07/eval/distill_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from common.jsonl import read_jsonl
from common.seed import seed_everything
from chapter05.metrics import token_f1

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant. Provide clear, step-by-step answers."
MAX_NEW_TOKENS = 256


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a single response using chat template."""
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_human_references(dolly_dir: str) -> dict:
    """Build a prompt -> original human answer map from the Dolly source
    (train + valid). The teacher prompts were drawn from this set, so every
    distillation prompt has a corresponding human reference."""
    refs = {}
    for fn in ("train.jsonl", "valid.jsonl"):
        path = Path(dolly_dir) / fn
        if not path.exists():
            continue
        for ex in read_jsonl(str(path)):
            msgs = ex.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user is not None and asst is not None:
                refs[user] = asst
    return refs


def evaluate_model(model, tokenizer, examples, label: str,
                   reference_mode: str = "human", human_refs: dict | None = None):
    """Run evaluation and return per-category token-F1.

    reference_mode="human": score against the original Dolly human answer
    (looked up by prompt). reference_mode="teacher": score against the
    teacher-generated assistant message in the example (old behavior)."""
    human_refs = human_refs or {}
    cat_f1 = defaultdict(list)
    missing = 0
    for ex in examples:
        msgs = ex["messages"]
        prompt = next(m["content"] for m in msgs if m["role"] == "user")
        if reference_mode == "human":
            reference = human_refs.get(prompt)
            if reference is None:
                # Fall back to the teacher output if no human ref is found,
                # but count it so we can flag incomplete coverage.
                reference = next(m["content"] for m in msgs if m["role"] == "assistant")
                missing += 1
        else:
            reference = next(m["content"] for m in msgs if m["role"] == "assistant")
        category = ex.get("category", "unknown")

        generated = generate_response(model, tokenizer, prompt)
        cat_f1[category].append(token_f1(generated, reference))
    if reference_mode == "human" and missing:
        print(f"  WARNING [{label}]: {missing} prompts had no human reference "
              f"(fell back to teacher output)")

    f1_by_cat = {
        cat: sum(scores) / len(scores)
        for cat, scores in sorted(cat_f1.items())
    }
    overall_f1 = (
        sum(s for scores in cat_f1.values() for s in scores)
        / max(sum(len(s) for s in cat_f1.values()), 1)
    )
    return f1_by_cat, overall_f1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate distillation: base vs teacher vs student"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--student_dir", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--dolly_dir", default="chapter06/data/dolly_sft",
                        help="Source of original human references (for --reference human)")
    parser.add_argument("--reference", choices=["human", "teacher"], default="human",
                        help="What to score against: original human answers (default) "
                             "or the teacher's own outputs (fidelity-to-teacher)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_everything(42)

    valid_path = Path(args.data_dir) / "valid.jsonl"
    examples = list(read_jsonl(str(valid_path)))
    print(f"Loaded {len(examples)} test examples")

    human_refs = {}
    if args.reference == "human":
        human_refs = load_human_references(args.dolly_dir)
        covered = sum(
            1 for ex in examples
            if next((m["content"] for m in ex["messages"] if m["role"] == "user"), None)
            in human_refs
        )
        print(f"Reference mode: HUMAN ({covered}/{len(examples)} prompts matched "
              f"to Dolly human answers in {args.dolly_dir})")
    else:
        print("Reference mode: TEACHER (fidelity-to-teacher; see module docstring)")

    results = {}

    # 1. Evaluate base model
    print("\n--- Evaluating base model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    base_f1, base_overall = evaluate_model(model, tokenizer, examples, "base", args.reference, human_refs)
    results["base"] = {"f1_by_category": base_f1, "overall_f1": base_overall}
    del model
    torch.cuda.empty_cache()

    # 2. Evaluate teacher (Ch6 SFT model)
    print("\n--- Evaluating teacher (Ch6 SFT) ---")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    teacher_f1, teacher_overall = evaluate_model(model, tokenizer, examples, "teacher", args.reference, human_refs)
    results["teacher"] = {"f1_by_category": teacher_f1, "overall_f1": teacher_overall}
    del model
    torch.cuda.empty_cache()

    # 3. Evaluate student (LoRA adapter)
    print("\n--- Evaluating student (distilled) ---")
    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.student_dir)
    model.eval()
    student_f1, student_overall = evaluate_model(model, tokenizer, examples, "student", args.reference, human_refs)
    results["student"] = {"f1_by_category": student_f1, "overall_f1": student_overall}

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Category':<25} {'Base':>8} {'Teacher':>8} {'Student':>8} {'Stud/Teach':>10}")
    print("-" * 70)
    all_cats = sorted(set(
        list(base_f1.keys()) + list(teacher_f1.keys()) + list(student_f1.keys())
    ))
    for cat in all_cats:
        b = base_f1.get(cat, 0.0)
        t = teacher_f1.get(cat, 0.0)
        s = student_f1.get(cat, 0.0)
        pct = f"{s / t:.0%}" if t > 0 else "N/A"
        print(f"{cat:<25} {b:>7.3f} {t:>8.3f} {s:>8.3f} {pct:>10}")
    print("-" * 70)
    pct_overall = f"{student_overall / teacher_overall:.0%}" if teacher_overall > 0 else "N/A"
    print(f"{'OVERALL':<25} {base_overall:>7.3f} {teacher_overall:>8.3f} "
          f"{student_overall:>8.3f} {pct_overall:>10}")
    print("=" * 70)

    # Save report
    if args.output:
        report = {
            "num_examples": len(examples),
            "reference_type": args.reference,
            "base_model": args.base_model,
            "teacher_dir": args.teacher_dir,
            "student_dir": args.student_dir,
            **{f"{k}_overall_f1": v["overall_f1"] for k, v in results.items()},
            **{f"{k}_f1_by_category": v["f1_by_category"] for k, v in results.items()},
            "student_as_pct_of_teacher": (
                student_overall / teacher_overall if teacher_overall > 0 else None
            ),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
