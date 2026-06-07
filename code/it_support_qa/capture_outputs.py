"""Capture base / base+prompt / adapter outputs on chosen golden questions
for the Ch5 prose (no retraining; loads the saved adapter)."""
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from it_support_qa.lora_domain_demo import BASE_SYS, PROMPT_SYS, generate, load

HERE = Path("it_support_qa")
BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = HERE / "runs/domain_lora"
PICK = [1, 3]  # network (GlobalConnect VPN), software (Software Center/StandardBuild)

golden = load(HERE / "data/expanded/eval.jsonl")
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
base.eval()
out = {}
for i in PICK:
    q = next(m["content"] for m in golden[i]["messages"] if m["role"] == "user")
    ref = next(m["content"] for m in golden[i]["messages"] if m["role"] == "assistant")
    out[i] = {"q": q, "ref": ref,
              "base": generate(base, tok, BASE_SYS, q),
              "base+prompt": generate(base, tok, PROMPT_SYS, q)}
adapter = PeftModel.from_pretrained(base, str(ADAPTER))
adapter.eval()
for i in PICK:
    q = out[i]["q"]
    out[i]["adapter"] = generate(adapter, tok, BASE_SYS, q)
(HERE / "results/sample_outputs.json").write_text(json.dumps(out, indent=2))
for i in PICK:
    print("="*70)
    print("Q:", out[i]["q"])
    for k in ("base", "base+prompt", "adapter"):
        print(f"\n----- {k} -----\n{out[i][k]}")
    print("\n----- reference (target) -----\n", out[i]["ref"])
