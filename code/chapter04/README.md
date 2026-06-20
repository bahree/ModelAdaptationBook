# Chapter 4 -- In-Context Learning, Few-Shot, and RAG

![Two axes of adaptation: prompting, few-shot, and RAG change the input; LoRA, SFT, and alignment change the model; hybrids combine both.](images/readme_two_axes.png)

*Two different ways to adapt a model. Prompting, few-shot, and RAG change what the model sees (the input axis); LoRA, SFT, and alignment change what the model is (the model axis). They are complementary, not competing, and most production systems combine both. This chapter covers the input-axis techniques.*

This chapter covers how to get useful work out of a model without training it: few-shot prompting, many-shot prompting on long-context models, prompt validation against held-out test sets, and a minimal retrieval-augmented generation (RAG) pipeline. The code in this folder backs the four numbered listings in the chapter.

The code is a mix of **CPU-friendly** modules (many-shot assembly, RAG, the prompt validator with the mock backend) and **GPU-friendly** modules (the full prompt validator and the few-shot demo with the transformers backend). Most of the chapter can be exercised on a laptop; the LLM-calling demos benefit from a GPU.

**Repository**: <https://github.com/bahree/ModelAdaptationBook>

### Where is the code?

All Chapter 4 code is in **this folder** (`code/chapter04/`):

| Location | What you'll find |
|----------|------------------|
| `*.py` (this folder) | Core modules: few-shot demo, many-shot assembly, prompt validator, minimal RAG. |
| `data/` | A small ticket-classification dataset and a policy-document corpus for the RAG examples. |
| `tests/` | Unit tests covering prompt assembly, similarity selection, the validator, and the RAG pipeline. |

Shared utilities (JSONL I/O, seeded reproducibility) live in **`code/common/`**.

**Chapter outline and listing map:**

| Listing | In the chapter | In the repo |
|---------|----------------|-------------|
| **4.1** | Few-shot prompt template | `few_shot_demo.py` |
| **4.2** | Many-shot prompt assembly | `many_shot_demo.py` |
| **4.3** | Prompt validator | `prompt_validator.py` |
| **4.4** | Minimal RAG pipeline | `rag_minimal.py` |
| **4.5** | RAG retrieval evaluation (Precision@k / Recall@k / Hit@1) | `scripts/listing_4_5_rag_eval.py` |

## Prerequisites

### One-time setup (fresh machine)

**First-time setup:** if you have not set up the book environment yet, follow the detailed instructions in **`code/README.md`** (one directory up). The chapter assumes:

- Python 3.10 or newer (3.12 recommended)
- A virtual environment with the book package installed: `pip install -e ".[dev]"`
- The `sentence-transformers` package for the default many-shot and RAG backends. It is installed automatically with the book package; if you skipped that step, run `pip install sentence-transformers`.

### GPU requirements

| Module | GPU needed | Notes |
|--------|-----------|-------|
| `few_shot_demo.py` (`--backend hf`) | Yes (~12 GB VRAM) | Loads `Qwen/Qwen3-4B-Instruct-2507` to classify tickets. |
| `few_shot_demo.py` (`--backend mock`) | No | Deterministic offline backend; useful for smoke tests. |
| `many_shot_demo.py` (`--backend st`) | No | Sentence-transformers on CPU. |
| `many_shot_demo.py` (`--backend hash`) | No | Token-overlap fallback; no extra dependencies. |
| `prompt_validator.py` (`--backend mock`) | No | Exercises the validator end to end without a model. |
| `prompt_validator.py` (`--backend hf`) | Yes (~12 GB VRAM) | Real validator runs on the Qwen3 model. |
| `rag_minimal.py` (`--backend st`) | No | Sentence-transformers + numpy index. |
| `rag_minimal.py` (`--backend hash`) | No | Hash-based fallback embedder. |

The book's reference setup is two NVIDIA A30 GPUs (24 GB each). Any GPU with 12 GB VRAM (RTX 3060/4060 or better) is enough for the `hf` backends.

### Verify your setup

```bash
# From code/ directory, venv activated
python -c "import chapter04; print('Chapter 4 imports OK')"

# Unit tests (CPU-only, no model required)
pytest chapter04/tests/ -v
```

## Step-by-step instructions

Run all commands below from the `code/` directory with your virtual environment activated.

### 1. Few-shot ticket classifier (Listing 4.1)

The few-shot demo builds an eight-shot prompt by sampling the example bank in `data/example_bank.jsonl` (stratified across categories), runs it against the held-out test set in `data/held_out_test.jsonl`, and reports top-1 accuracy.

**CPU smoke test (mock backend):**

```bash
python -m chapter04.few_shot_demo \
    --shots 8 --backend mock \
    --output chapter04/runs/few_shot_mock.json
```

**GPU run (Qwen3-4B):**

```bash
python -m chapter04.few_shot_demo \
    --shots 8 --backend hf \
    --output chapter04/runs/few_shot_hf.json
```

The script writes a summary JSON and a JSONL of per-example predictions.

### 2. Many-shot prompt assembly (Listing 4.2)

The many-shot demo selects the `k` examples whose ticket text is most similar to the incoming query, then prints the assembled prompt and an estimated token count.

```bash
# Sentence-transformers (default, CPU):
python -m chapter04.many_shot_demo \
    --query "I cannot log in after the password reset email" \
    --shots 20

# Token-overlap fallback (no embedding model):
python -m chapter04.many_shot_demo \
    --query "Refund my duplicate charge" \
    --shots 10 --backend hash
```

### 3. Prompt validator (Listing 4.3)

The validator runs the chosen prompt against the held-out set multiple times and reports mean accuracy, per-run accuracy, the disagreement rate (fraction of test cases where runs disagree), and per-example consistency.

```bash
# CPU smoke test:
python -m chapter04.prompt_validator \
    --shots 8 --runs 3 --backend mock

# GPU run:
python -m chapter04.prompt_validator \
    --shots 8 --runs 5 --backend hf \
    --output chapter04/runs/validator_hf.json
```

### 4. Minimal RAG pipeline (Listing 4.4)

The RAG demo ingests the policy documents in `data/policy_docs.jsonl`, builds a sentence-transformers vector index in numpy, and answers a query with retrieved context. The default LLM is a deterministic stub that echoes the prompt; replace `stub_llm` with a real model for production use.

```bash
# Inspect retrieval (no LLM call):
python -m chapter04.rag_minimal retrieve \
    --query "How do I rotate my API key?" --k 3

# Full retrieval + stub answer:
python -m chapter04.rag_minimal answer \
    --query "What is the API key rotation grace period?" --k 3

# Token-overlap fallback (no sentence-transformers required):
python -m chapter04.rag_minimal retrieve \
    --query "refund policy" --k 2 --backend hash
```

### 5. Measure RAG quality (Listing 4.5)

The RAG eval script runs a small labelled query set against the same pipeline and reports Precision@k, Recall@k, and Hit@1 — the retrieval-side metrics that let you tell whether a bad answer is the index's fault or the generator's. The labelled set in `data/rag_eval.jsonl` ships with one query per source document.

```bash
# Sentence-transformers backend (recommended):
python -m chapter04.scripts.listing_4_5_rag_eval --k 3

# Token-overlap fallback (CPU, no model download):
python -m chapter04.scripts.listing_4_5_rag_eval --k 3 --backend hash
```

The full report (per-query and aggregate) is written to `chapter04/runs/rag_eval.json`.

## Tests

```bash
pytest chapter04/tests/ -v
```

Tests cover the prompt assembly path, the similarity selector, the validator's accuracy and disagreement math, and the RAG chunker, retriever, and metadata flow. All tests run on CPU using deterministic mock backends.

## Troubleshooting

- **`No module named 'chapter04'`**: activate the virtual environment and run `pip install -e ".[dev]"` from `code/`. Then run commands from `code/` (the working directory matters).
- **`sentence-transformers` is not installed**: install it with `pip install sentence-transformers`, or use the `--backend hash` fallback for the many-shot and RAG demos.
- **`hf` backend exhausts GPU memory**: try a smaller model (`--model meta-llama/Llama-3.2-3B-Instruct`) or fall back to the `mock` backend for smoke testing.
- **The mock backend disagrees with my real model**: that is expected. The mock backend is a deterministic keyword classifier used for testing, not a quality reference.
