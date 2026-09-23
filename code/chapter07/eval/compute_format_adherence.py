"""Compute house-format adherence (the **Summary:/Steps:** signal) for the
three distillation models on the IT-support valid set.

Ch7's distill_report.json carries only token-F1, which is a blind proxy: it
cannot see whether a model reproduced the teacher's house format. This script
adds the behavioral lens. It reuses eval_3lens.py's IT_SYSTEM prompt,
format_ok() regex, and greedy generation, loading one model at a time and
freeing it before the next.

base    = Qwen/Qwen3-4B-Instruct-2507
teacher = chapter06/runs/sft_run1   (full SFT model)
student = chapter07/runs/student_run1 (LoRA adapter on base)

Run from code/:
    python -m chapter07.eval.compute_format_adherence
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common.env  # noqa: F401
from common.jsonl import read_jsonl

# Reuse the exact IT_SYSTEM prompt, format_ok() regex, and greedy gen settings
# from scripts/eval_3lens.py so the format lens matches the chapter's other eval.
from scripts.eval_3lens import BASE, IT_SYSTEM, format_ok, gen_all

TEACHER_DIR = "chapter06/runs/sft_run1"
STUDENT_DIR = "chapter07/runs/student_run1"
VALID = "data/it_support/valid.jsonl"
OUT = "chapter07/eval/format_report.json"
# --eval_file / --out override these (the book reports on data/it_support/test.jsonl, 2026-09-19)
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--eval_file", default=None); _ap.add_argument("--out", default=None)
_args, _ = _ap.parse_known_args()
if _args.eval_file: VALID = _args.eval_file
if _args.out: OUT = _args.out


def is_adapter(path: str) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def load(model_path: str, adapter: str | None = None):
    m = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0"
    )
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m


def fmt_pct(model, tok, prompts) -> tuple[float, int, int]:
    gens = gen_all(model, tok, prompts)
    ok = sum(format_ok(g) for g in gens)
    return round(ok / len(gens), 4), ok, len(gens)


def main():
    val = list(read_jsonl(VALID))
    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = [
        [
            {"role": "system", "content": IT_SYSTEM},
            {"role": "user", "content": next(m["content"] for m in ex["messages"] if m["role"] == "user")},
        ]
        for ex in val
    ]
    print(f"Loaded {len(prompts)} prompts from {VALID}")

    report = {
        "num_examples": len(prompts),
        "valid_set": VALID,
        "system_prompt": IT_SYSTEM,
        "format_regex": r"**Summary:** ... **Steps:** ... 1.",
        "base_model": BASE,
        "teacher_dir": TEACHER_DIR,
        "student_dir": STUDENT_DIR,
        "metric": "house_format_adherence (fraction matching **Summary:/Steps:** regex)",
    }

    # base
    print("generating base...", flush=True)
    m = load(BASE)
    pct, ok, n = fmt_pct(m, tok, prompts)
    report["base_format_adherence"] = pct
    report["base_format_count"] = f"{ok}/{n}"
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  base format {pct*100:.0f}% ({ok}/{n})")

    # teacher (full model)
    print("generating teacher...", flush=True)
    m = load(TEACHER_DIR)
    pct, ok, n = fmt_pct(m, tok, prompts)
    report["teacher_format_adherence"] = pct
    report["teacher_format_count"] = f"{ok}/{n}"
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  teacher format {pct*100:.0f}% ({ok}/{n})")

    # student (adapter on base, per task spec check)
    print("generating student...", flush=True)
    if is_adapter(STUDENT_DIR):
        print(f"  {STUDENT_DIR} is a LoRA adapter -> loading base + PeftModel")
        m = load(BASE, adapter=STUDENT_DIR)
    else:
        print(f"  {STUDENT_DIR} is a full model -> loading directly")
        m = load(STUDENT_DIR)
    pct, ok, n = fmt_pct(m, tok, prompts)
    report["student_format_adherence"] = pct
    report["student_format_count"] = f"{ok}/{n}"
    del m
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  student format {pct*100:.0f}% ({ok}/{n})")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
