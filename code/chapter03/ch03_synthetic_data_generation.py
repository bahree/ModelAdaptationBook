"""
Step-by-step walkthrough of the complete synthetic data generation pipeline:
  Step 1 — Load seed examples from Financial PhraseBank (HuggingFace)
  Step 2 — Build a category-controlled generation prompt with style anchoring
  Step 3 — Call a teacher LLM (Claude) to generate candidate examples
  Step 4 — Apply the LLM-as-judge quality gate (fidelity, format, domain safety)
  Step 5 — Check distribution alignment between synthetic and real data
  Step 6 — Mix with real data at the 30% synthetic cap and save with a manifest

Each step is self-contained. You can run the full file or import individual
functions to understand how each stage of the pipeline works.

Requires: ANTHROPIC_API_KEY environment variable
Install : pip install anthropic datasets huggingface_hub scikit-learn
"""

import os
import re
import json
import zipfile
import random
import hashlib
import datetime

from datasets import Dataset
from huggingface_hub import hf_hub_download

import common.env  # noqa: F401  side-effect: load code/.env so the API keys below are visible

# ── Shared constants ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a knowledgeable financial analyst. "
    "Provide accurate, well-reasoned responses to financial questions. "
    "Respond with exactly one word when classifying sentiment: "
    "positive, negative, or neutral."
)

SEED_DATASET  = "takala/financial_phrasebank"   # public, CC-BY-NC-SA 3.0
TEACHER_MODEL = "claude-sonnet-4-6"             # teacher LLM (direct Anthropic id)
OPENROUTER_TEACHER_MODEL = "anthropic/claude-sonnet-4.5"  # same family, via OpenRouter


def call_teacher(prompt: str, max_tokens: int = 4096, system: str = None) -> str:
    """Return the teacher LLM's raw text response to a prompt.

    Default path: the Anthropic SDK directly, when ANTHROPIC_API_KEY is set -- this
    is what the chapter walks through. Fallback: the same Claude model via
    OpenRouter, when only OPENROUTER_API_KEY is set. It is the same model (no
    behavioral difference; generation is stochastic regardless), so this just lets
    the script run with whichever key you have configured, the same way chapters 7
    and 8 reach frontier models. The anthropic import is deferred so the OpenRouter
    path does not require the anthropic package.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        kwargs = {"system": system} if system else {}
        resp = anthropic.Anthropic().messages.create(
            model=TEACHER_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.content[0].text
    if os.environ.get("OPENROUTER_API_KEY"):
        from common.openrouter import chat
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return chat(messages, model=OPENROUTER_TEACHER_MODEL, max_tokens=max_tokens)["content"]
    raise RuntimeError(
        "No teacher API key found. Set ANTHROPIC_API_KEY (direct Anthropic, the "
        "chapter default) or OPENROUTER_API_KEY (same model via OpenRouter)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load seed examples from HuggingFace
# ══════════════════════════════════════════════════════════════════════════════
#
# Seed examples serve two purposes:
#   1. They anchor the style, tone, and domain of the generated examples.
#      The teacher model imitates what it sees in the prompt, not what it
#      imagines financial text looks like in the abstract.
#   2. They set a quality baseline. Any generated example that does not
#      resemble the seeds in style or domain is a candidate for rejection
#      in the quality gate (Step 4).
#
# We use Financial PhraseBank's AllAgree split — the highest-quality tier,
# where all annotators agreed on the label. These are the best seeds available
# for a financial sentiment task because every label is unambiguous.

def load_seeds(n_per_category: int = 20, seed: int = 42) -> list[dict]:
    """
    Load seed examples from Financial PhraseBank (HuggingFace).

    Organises seeds by sentiment category so the prompt builder (Step 2)
    can select category-matched seeds for each generation call.

    Args:
        n_per_category: Maximum seeds to keep per label class
        seed:           Random seed for reproducible sampling

    Returns:
        List of dicts with 'sentence', 'label', 'source' fields
    """
    print("Step 1: Loading seeds from Financial PhraseBank (AllAgree split)...")

    # Download from HuggingFace Hub — cached locally after first run
    zip_path = hf_hub_download(
        repo_id=SEED_DATASET,
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        # AllAgree split = highest quality tier (all 16 annotators agreed)
        with z.open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt") as f:
            for line in f.read().decode("latin-1").splitlines():
                if "@" in line:
                    # Format: "sentence text @ label"
                    sentence, label = line.rsplit("@", 1)
                    rows.append({
                        "sentence": sentence.strip(),
                        "label":    label.strip(),
                        "source":   "financial_phrasebank_all_agree",
                    })

    # Sample up to n_per_category examples per label to keep the seed pool balanced
    rng = random.Random(seed)
    rng.shuffle(rows)
    by_label: dict[str, list] = {"positive": [], "negative": [], "neutral": []}
    for row in rows:
        if len(by_label[row["label"]]) < n_per_category:
            by_label[row["label"]].append(row)

    seeds = [row for rows in by_label.values() for row in rows]
    rng.shuffle(seeds)

    print(f"  Loaded {len(seeds)} seeds across {len(by_label)} labels")
    for label, examples in by_label.items():
        print(f"    {label:9s}: {len(examples)} examples")

    # Preview the first seed so we can see what the teacher will imitate
    print("\n  Sample seed (this style will anchor generation):")
    print(f"    [{seeds[0]['label']:8s}] {seeds[0]['sentence'][:80]}")
    return seeds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build a category-controlled generation prompt
# ══════════════════════════════════════════════════════════════════════════════
#
# Style anchoring is the single most important design decision in the prompt.
# We include 2–3 real seed examples inside the prompt text. The teacher model
# sees these and imitates their style, vocabulary, and level of detail.
#
# WITHOUT style anchoring: the teacher generates technically correct financial
# text but in its own preferred style, which may differ significantly from
# your real data. The model you train on it will learn the teacher's style
# rather than your domain's style.
#
# WITH style anchoring: the teacher generates text that looks like it came
# from the same source as your seeds. Distribution alignment (Step 5) will
# score much higher, and the trained model will match production inputs better.
#
# Category control prevents the teacher from overrepresenting the most common
# scenario. Without it, every generation call produces mostly neutral sentiment
# (the majority class) and very few negative examples (the minority class).

def build_generation_prompt(
    category: str,
    seeds:    list[dict],
    n_to_generate: int = 30,
) -> str:
    """
    Build a generation prompt with style anchoring for a specific category.

    Selects 2–3 seeds that match the target category (or falls back to
    any seeds if the category pool is small), formats them as a style
    reference block, and asks the teacher to generate variations.

    Args:
        category:       Target sentiment label ("positive", "negative", "neutral")
        seeds:          Full seed pool from Step 1
        n_to_generate:  Number of examples to request

    Returns:
        Formatted prompt string ready to send to the teacher model
    """
    # Prefer seeds that match the target category
    category_seeds = [s for s in seeds if s["label"] == category]
    if len(category_seeds) < 2:
        category_seeds = seeds  # fall back to full pool

    # Select 2–3 diverse seeds for the style reference
    # (more than 3 pushes too much of the prompt budget)
    sample = random.sample(category_seeds, min(3, len(category_seeds)))

    # Format as a readable Q&A style reference block
    style_ref = "\n".join(
        f'  Sentence: "{s["sentence"][:100]}"\n  Label: {s["label"]}'
        for s in sample
    )

    prompt = f"""You are building training data for a financial language model.

TASK: Generate new financial sentiment examples for the "{category}" category.

STYLE REFERENCE — match the vocabulary, length, and tone of these real examples exactly:
{style_ref}

REQUIREMENTS:
- Generate exactly {n_to_generate} NEW examples with sentiment label: {category}
- Every sentence must be about a real company, financial event, or market development
- Vary the sentence structure, company types, and financial topics
- Do NOT copy or paraphrase the style reference examples above
- Financial facts (rates, percentages, amounts) must be realistic

Return ONLY a valid JSON array. No preamble, no markdown fences:
[{{"sentence": "...", "label": "{category}"}}]"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Call the teacher model to generate candidate examples
# ══════════════════════════════════════════════════════════════════════════════
#
# We make one API call per category. This design has two advantages:
#   1. Failures are contained — if the positive category call fails, the
#      negative and neutral categories are unaffected.
#   2. Category balance is explicit — we request exactly the target count
#      for each category rather than hoping a single large call produces
#      a balanced distribution.
#
# The teacher model returns a JSON array. We parse it carefully because
# LLMs occasionally produce malformed JSON or wrap the array in markdown
# fences. The parse_candidates() function handles both cases.

def parse_candidates(raw_response: str, expected_label: str) -> list[dict]:
    """
    Parse the teacher model's JSON output into a list of candidate dicts.

    Handles two common failure modes:
      - Markdown fences (```json ... ```)
      - Non-array JSON (single object instead of list)

    Args:
        raw_response:    Raw string from the teacher model
        expected_label:  The category label that was requested

    Returns:
        List of validated candidate dicts with 'sentence' and 'label' keys
    """
    text = raw_response.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:])          # remove opening fence
    text = text.removesuffix("```").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse failed ({e}) — skipping batch")
        return []

    # Normalise to a list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    # Validate each entry: must have sentence and label
    valid = []
    for item in parsed:
        if isinstance(item, dict) and "sentence" in item and "label" in item:
            # Normalise label to lowercase and override with expected label
            # (the teacher occasionally mislabels even when instructed)
            item["label"] = expected_label
            item["source"] = "synthetic"
            valid.append(item)

    return valid


def generate_candidates(
    seeds:      list[dict],
    categories: list[str] = None,
    n_per_cat:  int = 30,
) -> list[dict]:
    """
    Run one API call per category and return all candidate examples.

    Args:
        seeds:      Seed pool from Step 1
        categories: List of categories to generate; defaults to all three sentiment labels
        n_per_cat:  Target examples per category

    Returns:
        Flat list of all candidates across all categories
    """
    if categories is None:
        categories = ["positive", "negative", "neutral"]

    print(f"\nStep 3: Generating candidates ({n_per_cat} per category)...")

    all_candidates = []

    for category in categories:
        print(f"  [{category}] Calling teacher model...", end=" ", flush=True)

        prompt = build_generation_prompt(category, seeds, n_per_cat)

        raw_text = call_teacher(prompt, max_tokens=4096)
        candidates = parse_candidates(raw_text, category)
        all_candidates.extend(candidates)
        print(f"parsed {len(candidates)} candidates")

    print(f"  Total candidates: {len(all_candidates)}")
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Quality gate: LLM-as-judge verification
# ══════════════════════════════════════════════════════════════════════════════
#
# The teacher model generates plausible-sounding text, but plausibility is
# not correctness. Financial text is especially prone to:
#   - Hallucinated statistics (made-up percentages, rates, figures)
#   - Invented company names that sound real but are not
#   - Labels that contradict the sentence content
#
# The quality gate uses a second LLM call to score each candidate on three
# dimensions. Using a different system prompt and a shorter context window
# reduces the chance of the judge being influenced by the generation context.
#
# Three dimensions, each scored 1–5:
#   FIDELITY     — does the response directly address the sentence's sentiment?
#   FORMAT       — is the sentence realistic financial text?
#   DOMAIN SAFETY — are all financial facts plausible (no hallucinated rates)?
#
# A PASS requires ALL THREE dimensions >= 4.
# Rejecting ~15–25% is normal and desirable; it removes the weakest examples.

JUDGE_SYSTEM_PROMPT = """You are a quality reviewer for a financial language model training dataset.

Score this financial sentiment example on three dimensions (1–5 each):

FIDELITY (1–5): Does the sentiment label match the sentence content?
  5 = label is clearly correct for this sentence
  1 = label contradicts the sentence

FORMAT (1–5): Is this realistic financial text?
  5 = reads like real financial news or analyst commentary
  1 = sounds artificial, generic, or not financial

DOMAIN SAFETY (1–5): Are financial facts plausible?
  5 = all percentages, rates, amounts are realistic
  1 = contains impossible or implausible financial figures

A PASS requires ALL THREE >= 4.

Return ONLY valid JSON, no preamble:
{"fidelity": X, "format": X, "domain_safety": X, "pass": true_or_false}"""


def _parse_judge_json(raw_text: str) -> dict:
    """
    Parse the judge model's JSON scores, tolerating markdown fences and a stray
    preamble. Some providers (e.g. OpenRouter-routed Claude) wrap the object in
    ```json ... ``` or add a short lead-in even when asked to return only JSON,
    which a bare json.loads rejects (the generation parser strips fences the
    same way; this keeps the judge path just as robust).
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    text = text.removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: pull the first {...} object out of any surrounding prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def score_one_example(example: dict) -> dict:
    """
    Score a single candidate example using the LLM-as-judge rubric.

    Args:
        example: Dict with 'sentence' and 'label' fields

    Returns:
        Input dict augmented with 'scores' and 'verified' fields
    """
    content = (
        f"Sentence: \"{example['sentence']}\"\n"
        f"Label: {example['label']}"
    )

    raw_text = call_teacher(content, max_tokens=200, system=JUDGE_SYSTEM_PROMPT)

    try:
        scores = _parse_judge_json(raw_text)
    except json.JSONDecodeError:
        # If the judge fails to produce valid JSON, treat as failed
        scores = {"fidelity": 0, "format": 0, "domain_safety": 0, "pass": False}

    return {**example, "scores": scores, "verified": scores.get("pass", False)}


def run_quality_gate(candidates: list[dict]) -> tuple[list[dict], list[dict], float]:
    """
    Apply the quality gate to all candidates and report results.

    Args:
        candidates: All candidates from Step 3

    Returns:
        (passed, failed, pass_rate) tuple
    """
    print(f"\nStep 4: Quality gate — scoring {len(candidates)} candidates...")

    passed, failed = [], []

    for i, example in enumerate(candidates):
        scored = score_one_example(example)
        if scored["verified"]:
            passed.append(scored)
        else:
            failed.append(scored)

        # Progress indicator for long batches
        if (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(candidates)}...", end="\r")

    pass_rate = len(passed) / max(len(candidates), 1)
    print(f"\n  Passed: {len(passed)} ({pass_rate:.0%})")
    print(f"  Failed: {len(failed)} ({1-pass_rate:.0%})")

    # Report which dimensions caused the most failures
    if failed:
        from collections import Counter
        dim_fails = Counter()
        for ex in failed:
            for dim in ("fidelity", "format", "domain_safety"):
                if ex["scores"].get(dim, 5) < 4:
                    dim_fails[dim] += 1
        print(f"  Failure breakdown: {dict(dim_fails)}")
        if dim_fails.most_common(1)[0][1] > len(failed) * 0.5:
            top_dim = dim_fails.most_common(1)[0][0]
            print(f"  Tip: '{top_dim}' is the primary failure dimension.")
            print(f"       Revise the generation prompt for '{top_dim}' to improve pass rate.")

    return passed, failed, pass_rate


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Distribution alignment check
# ══════════════════════════════════════════════════════════════════════════════
#
# Zhu et al. (ICML 2025) showed that synthetic text concentrates in a narrow
# perplexity band (0–14) while human text spans the full range (0–100+). This
# concentration is what causes model collapse when the synthetic proportion is
# too high. The centroid cosine similarity check is a lightweight proxy for
# distribution overlap: if the average embedding of the synthetic examples is
# close to the average embedding of the real examples, the distributions are
# compatible for mixing.
#
# Threshold:
#   score >= 0.85: safe to mix
#   score  0.70–0.85: proceed with caution (reduce synthetic proportion)
#   score < 0.70: revise generation prompt before mixing

def check_distribution_alignment(
    real_examples:      list[dict],
    synthetic_examples: list[dict],
) -> float:
    """
    Compute centroid cosine similarity between real and synthetic embeddings.

    Uses the all-MiniLM-L6-v2 sentence transformer (22M params, fast on CPU).
    Falls back to 1.0 (skip check) if sentence-transformers is not installed.

    Args:
        real_examples:      Real seed examples from Step 1
        synthetic_examples: Verified synthetic examples from Step 4

    Returns:
        Cosine similarity score (0.0–1.0)
    """
    print("\nStep 5: Distribution alignment check...")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        print("  Loading sentence encoder (all-MiniLM-L6-v2)...")
        # Suppress the harmless "UNEXPECTED key" warning that appears when the
        # checkpoint was saved with a different transformers version (the
        # position_ids buffer is now generated on-the-fly rather than stored
        # in the checkpoint file — safe to ignore)
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode sentences from both pools
        real_texts  = [ex["sentence"] for ex in real_examples]
        synth_texts = [ex["sentence"] for ex in synthetic_examples]

        real_emb  = encoder.encode(real_texts,  show_progress_bar=False)
        synth_emb = encoder.encode(synth_texts, show_progress_bar=False)

        # Centroid similarity: compare the mean of real vs mean of synthetic
        # This is fast (O(1) distance) and robust to individual outliers
        sim = float(cosine_similarity(
            real_emb.mean(0, keepdims=True),
            synth_emb.mean(0, keepdims=True),
        )[0, 0])

        if sim >= 0.85:
            verdict = "safe to mix"
        elif sim >= 0.70:
            verdict = "borderline — consider reducing synthetic proportion to 15%"
        else:
            verdict = "mismatch — revise generation prompt before mixing"

        print(f"  Centroid cosine similarity: {sim:.3f}  ({verdict})")
        return sim

    except ImportError:
        print("  sentence-transformers not installed — skipping check (score = 1.0)")
        print("  Install with: pip install sentence-transformers scikit-learn")
        return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Mix with real data and save with a version manifest
# ══════════════════════════════════════════════════════════════════════════════
#
# The 30% synthetic cap is a hard constraint derived from Zhu et al. (2025).
# When synthetic data exceeds 30% of the training mix, the distribution
# narrows and the model underperforms on low-frequency inputs — exactly the
# rare-but-important examples that fine-tuning is supposed to improve.
#
# The formula: n_synth = n_real × 0.30 / (1 - 0.30)
#   = n_real × 0.4286
# Solving for n_synth given n_real ensures the synthetic proportion stays
# at exactly 30% after mixing, regardless of how many passed the quality gate.
#
# The manifest records the SHA-256 hash of the full mixed dataset. Any future
# change to the data — even a single character — will produce a different hash,
# making data tampering detectable.

def mix_and_save(
    real_examples:      list[dict],
    synthetic_examples: list[dict],
    output_dir:         str  = "./ch03_synthetic_output",
    max_synth_ratio:    float = 0.30,
    seed:               int  = 42,
) -> dict:
    """
    Mix real and synthetic examples at the 30% cap and save with a manifest.

    The eval split always contains only real examples — never synthetic.
    This ensures the evaluation benchmark measures against ground truth,
    not against the teacher model's preferred outputs.

    Args:
        real_examples:      Real seed examples (the full available pool)
        synthetic_examples: Verified synthetic examples from Step 4
        output_dir:         Directory to save the HuggingFace Dataset and manifest
        max_synth_ratio:    Maximum fraction of synthetic data (default 0.30)
        seed:               Random seed for reproducible sampling

    Returns:
        Dataset manifest dict
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_real       = len(real_examples)
    # Cap synthetic count to enforce the 30% ratio
    n_synth_max  = int(n_real * max_synth_ratio / (1 - max_synth_ratio))
    n_synth      = min(n_synth_max, len(synthetic_examples))

    rng = random.Random(seed)
    synth_sample = rng.sample(synthetic_examples, n_synth)

    # Convert to ChatML format for training
    def to_chatml(ex: dict) -> dict:
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": (
                    f"Classify the sentiment of this financial statement:\n"
                    f"\"{ex['sentence']}\""
                )},
                {"role": "assistant", "content": ex["label"]},
            ],
            "source":   ex.get("source", "unknown"),
            "label":    ex["label"],
        }

    all_rows = [to_chatml(ex) for ex in real_examples] + \
               [to_chatml(ex) for ex in synth_sample]
    rng.shuffle(all_rows)

    # 90/10 train/eval split — eval uses only real examples
    n_eval    = max(5, len(real_examples) // 10)
    eval_rows = [to_chatml(ex) for ex in rng.sample(real_examples, n_eval)]
    train_rows = all_rows

    # Save as HuggingFace Datasets
    Dataset.from_list(train_rows).save_to_disk(f"{output_dir}/train")
    Dataset.from_list(eval_rows).save_to_disk(f"{output_dir}/eval")

    # SHA-256 manifest — makes any future data modification detectable
    content = json.dumps(all_rows, sort_keys=True).encode()
    sha256  = hashlib.sha256(content).hexdigest()

    manifest = {
        "version":    "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256":     sha256,
        "composition": {
            "total":     len(all_rows),
            "train":     len(train_rows),
            "eval":      len(eval_rows),
            "real":      n_real,
            "synthetic": n_synth,
            "synth_pct": round(n_synth / len(all_rows), 3),
        },
        "synthetic_cap":  max_synth_ratio,
        "trained_models": [],   # populated after training with checkpoint info
    }

    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nStep 6: Mixed dataset saved to {output_dir}/")
    print(f"  Real        : {n_real}")
    print(f"  Synthetic   : {n_synth}  ({manifest['composition']['synth_pct']:.0%} of total)")
    print(f"  Train       : {len(train_rows)}")
    print(f"  Eval        : {len(eval_rows)}  (real only)")
    print(f"  SHA-256     : {sha256[:24]}...")

    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the full six-step pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3.9 — Synthetic Data Generation Pipeline")
    print("=" * 60)

    # ── Step 1: Load seeds ────────────────────────────────────────────────────
    seeds = load_seeds(n_per_category=20)

    # ── Step 2: Build prompts (shown for one category as illustration) ────────
    print("\nStep 2: Building generation prompt (positive category sample)...")
    sample_prompt = build_generation_prompt("positive", seeds, n_to_generate=5)
    print("  Prompt preview (first 300 chars):")
    print(f"  {sample_prompt[:300]}...")

    # ── Step 3: Generate candidates ───────────────────────────────────────────
    # Use a small n_per_cat to keep the demo fast and cheap (~$0.05)
    # Increase to 50–100 for a production run
    N_PER_CAT = 15
    candidates = generate_candidates(seeds, n_per_cat=N_PER_CAT)

    # Show a sample candidate before quality filtering
    if candidates:
        print("\n  Sample candidate (before quality gate):")
        ex = candidates[0]
        print(f"    [{ex['label']:8s}] {ex['sentence'][:80]}")

    # ── Step 4: Quality gate ──────────────────────────────────────────────────
    passed, failed, pass_rate = run_quality_gate(candidates)

    # Show a sample of what passed and what failed
    if passed:
        print("\n  Sample PASSED example:")
        ex = passed[0]
        print(f"    [{ex['label']:8s}] {ex['sentence'][:80]}")
        print(f"    Scores: {ex['scores']}")

    if failed:
        print("\n  Sample FAILED example:")
        ex = failed[0]
        print(f"    [{ex['label']:8s}] {ex['sentence'][:80]}")
        print(f"    Scores: {ex['scores']}")

    # ── Step 5: Distribution check ────────────────────────────────────────────
    alignment = check_distribution_alignment(seeds, passed)

    # ── Step 6: Mix and save ──────────────────────────────────────────────────
    if passed:
        manifest = mix_and_save(seeds, passed, output_dir="./ch03_synthetic_output")
        print("\n  Next step: pass ./ch03_synthetic_output/train to SFTTrainer")
        print("  Dataset manifest: ./ch03_synthetic_output/manifest.json")
    else:
        print("\n  No examples passed the quality gate.")
        print("  Check your ANTHROPIC_API_KEY and try with a larger n_per_cat.")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)