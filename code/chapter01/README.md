# Chapter 1: Why Model Adaptation?

![The model adaptation continuum: prompting and RAG, LoRA/QLoRA, full SFT, distillation, and alignment, with effort, GPU need, and cost rising from left to right.](../../assets/adaptation-continuum.png)

*The mental model for the whole book. The techniques in chapters 4 to 8 trade rising engineering effort, GPU requirements, and cost for greater influence over the model's behavior. Chapters 1 to 3 set up the why, how, and data, and chapter 9 keeps the result reliable in production. This chapter explains where each technique fits and how to choose between them.*

Chapter 1 is a framing chapter: it sets up *why* and *when* to adapt LLMs and lays out the decision framework and continuum that organize the rest of the book. There is no training run in this chapter and no model to ship at the end of it.

The one piece of code here exists so the §1.6 sidebar ("What the gap actually looks like") is reproducible: any reader with a working environment can run the script and confirm the outputs the chapter prints.

## What you need

- The repo setup from [`code/README.md`](../README.md): Python 3.10+, PyTorch with CUDA, `pip install -e ".[dev]"`.
- A GPU. Qwen3-4B-Instruct-2507 needs ~8 GB of VRAM in bf16 (any consumer card 12 GB and up). CPU works too, but inference takes minutes per response instead of seconds.
- Optional, only for the second and third configurations:
  - The LoRA adapter produced by Chapter 5 at `chapter05/runs/it_lora/`.
  - The full SFT model produced by Chapter 6 at `chapter06/runs/sft_run1/`.

If you have not run Chapters 5 and 6 yet, the script still runs the base model and prints a clear note for the missing configurations. You can come back and re-run after producing those artifacts.

## Running

From the `code/` directory with your venv activated:

```bash
python -m chapter01.run_sidebar_example
```

Output goes to stdout (each model's response is printed under a labeled header) and to `chapter01/sidebar_outputs.json` for reference.

To run only the base model and skip the optional configurations:

```bash
python -m chapter01.run_sidebar_example --base_only
```

To point at adapters/models in non-default locations:

```bash
python -m chapter01.run_sidebar_example \
    --lora_dir /path/to/my/lora_adapter \
    --sft_dir  /path/to/my/sft_model
```

## What you should see

The base model (Qwen3-4B-Instruct-2507) will confidently invent a deadline for the fictional reimbursement policy and add a closing hedge sentence ("please refer to the official policy document").

The LoRA-tuned and SFT-tuned variants from Chapters 5 and 6 will also confidently invent a deadline, but they will *strip* the closing hedge, because their training on the book's IT-support dataset (real Stack Exchange IT Q&A plus a small Dolly mix-in, built by `code/scripts/build_it_support_dataset.py`) taught them to answer instructions directly rather than defer.

That is the point of the sidebar. Adaptation on generic instruction-following data does not fix confabulation; it can make a wrong answer more confident. Chapter 3 covers the data strategy that actually fixes this; Chapter 8 covers the preference-optimization techniques that shape refusal behavior.

The exact text the chapter quotes was captured on an NVIDIA A30 with bf16 and greedy decoding (deterministic). Your outputs should match word for word on the same hardware and library versions, and should be very close on similar hardware.

---

**Repository:** <https://github.com/bahree/ModelAdaptationBook>
