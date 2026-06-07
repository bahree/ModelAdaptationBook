"""Domain-LoRA demo: show that LoRA beats prompting on Contoso-internal Q&A.

Trains a LoRA adapter on the expanded Contoso IT-support set, then compares three
configurations on the held-out golden questions:
  - base            : Qwen3-4B-Instruct, no adapter, minimal system prompt
  - base + prompt   : same model, a strong FORMAT system prompt (house format,
                      but NOT the internal tool glossary, which you would not put
                      in every prompt and which the model has never seen)
  - adapter         : the LoRA-trained model

The headline metric is internal-terminology use: the adapter learns Contoso's
names (GlobalConnect VPN, AccessHub, StandardBuild, #it-help); base and
base+prompt cannot, because no prompt supplied them. That is the gain a prompt
cannot replicate. We also report house-format compliance and token-F1.

Needs a GPU + trl/peft/transformers (same stack as Chapter 5).
Run from code/:  python3 -m it_support_qa.lora_domain_demo --max-steps 120
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BASE_SYS = "You are Contoso's internal IT support assistant. Give clear, accurate help."
# Realistic "prompt baseline": ask for the house FORMAT, but do not hand the model
# the internal tool glossary (you would not paste it into every request, and it
# still would not know which internal tool applies).
PROMPT_SYS = (
    "You are an enterprise IT support assistant. Answer in this format: one direct "
    "sentence, then a line 'Steps:' with 2 to 4 numbered steps, then a final line "
    "telling the user how to escalate if it does not work. Be concise.")
# body vocabulary only (excludes the formulaic escalation terms #it-help /
# ServiceNow, which every house-style answer ends with) so the metric measures
# real domain-tool recall, not the learned escalation line.
INTERNAL_TERMS = ["GlobalConnect", "AccessHub", "StandardBuild",
                  "Contoso MFA", "Software Center"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def token_f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def uses_internal_term(text):
    return any(t.lower() in text.lower() for t in INTERNAL_TERMS)


def format_ok(text):
    return "Steps:" in text and ("#it-help" in text or "ServiceNow" in text)


def train_lora(base, train_path, out_dir, max_steps):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    rows = load(train_path)
    tok = AutoTokenizer.from_pretrained(base)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    cfg = SFTConfig(output_dir=str(out_dir), num_train_epochs=3,
                    per_device_train_batch_size=2, gradient_accumulation_steps=4,
                    learning_rate=2e-4, max_steps=max_steps, logging_steps=10,
                    warmup_ratio=0.1, report_to=[])
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                    processing_class=tok)
    tr.train()
    tr.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del tr, model
    torch.cuda.empty_cache()


def generate(model, tok, system, question):
    import torch
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=220, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def score(model, tok, system, golden):
    term = fmt = 0
    f1s = []
    for ex in golden:
        q = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        ref = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        ans = generate(model, tok, system, q)
        term += uses_internal_term(ans)
        fmt += format_ok(ans)
        f1s.append(token_f1(ans, ref))
    n = len(golden)
    return {"internal_tool_in_body": round(term / n, 3), "house_format": round(fmt / n, 3),
            "token_f1": round(sum(f1s) / n, 3), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train", default=str(HERE / "data/expanded/train.jsonl"))
    ap.add_argument("--eval", default=str(HERE / "data/expanded/eval.jsonl"))
    ap.add_argument("--out", default=str(HERE / "runs/domain_lora"))
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    golden = load(args.eval)
    print(f"training LoRA on {len(load(args.train))} examples ...")
    train_lora(args.base, args.train, Path(args.out), args.max_steps)

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                device_map="auto")
    base.eval()
    res = {"base": score(base, tok, BASE_SYS, golden),
           "base+prompt": score(base, tok, PROMPT_SYS, golden)}
    adapter = PeftModel.from_pretrained(base, args.out)
    adapter.eval()
    res["adapter"] = score(adapter, tok, BASE_SYS, golden)

    outp = Path(args.out) / "results.json"
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nExpected: internal_term_use base~0, base+prompt~0, adapter high "
          "(the gain a prompt cannot supply).")


if __name__ == "__main__":
    main()
