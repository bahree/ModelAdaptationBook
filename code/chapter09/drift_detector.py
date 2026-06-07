"""Listing 9.2 -- Embedding-based drift detection using TF-IDF + cosine similarity.

Compares a *reference* prompt distribution (e.g. training data) against a
*production* prompt distribution (e.g. recent user queries) to detect
when inputs have drifted enough to warrant investigation or retraining.

Uses a lightweight TF-IDF vectoriser implemented with numpy -- no GPU
and no external ML libraries required.  For production systems, replace
the TF-IDF step with sentence-transformer embeddings or the model's own
encoder representations.

Alert thresholds:
    drift < 0.10  ->  GREEN   (no action)
    drift 0.10-0.20  ->  YELLOW  (investigate)
    drift > 0.20  ->  RED     (consider retraining)

Run from code/:
    python -m chapter09.drift_detector \
        --reference chapter06/data/dolly_sft/train.jsonl \
        --production chapter09/data/sample_production.jsonl \
        --output chapter09/eval/drift_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common.jsonl import read_jsonl


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
DRIFT_GREEN = 0.10
DRIFT_YELLOW = 0.20


def alert_level(drift_score: float) -> str:
    """Map a drift score to an alert level.

    Args:
        drift_score: Non-negative drift score (0 = identical distributions).

    Returns:
        One of ``"green"``, ``"yellow"``, or ``"red"``.
    """
    if drift_score < DRIFT_GREEN:
        return "green"
    if drift_score < DRIFT_YELLOW:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Lightweight TF-IDF (no sklearn needed)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_tfidf_matrix(documents: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Build a TF-IDF matrix from a list of documents.

    Args:
        documents: List of text strings.

    Returns:
        Tuple of (tfidf_matrix, vocabulary) where tfidf_matrix has shape
        ``(n_docs, n_vocab)`` and vocabulary is the sorted token list.
    """
    # Build vocabulary across all documents
    doc_tokens = [_tokenize(doc) for doc in documents]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    vocab = sorted(doc_freq.keys())
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_docs = len(documents)
    n_vocab = len(vocab)

    if n_vocab == 0:
        return np.zeros((n_docs, 1)), []

    # Compute TF-IDF
    matrix = np.zeros((n_docs, n_vocab), dtype=np.float64)
    for doc_i, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        for token, count in tf.items():
            if token in token_to_idx:
                col = token_to_idx[token]
                # TF: log-normalised, IDF: smooth inverse document frequency
                tf_val = 1 + math.log(count) if count > 0 else 0
                idf_val = math.log(1 + n_docs / (1 + doc_freq[token]))
                matrix[doc_i, col] = tf_val * idf_val

    # L2-normalise rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    return matrix, vocab


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Returns:
        Float in [-1, 1].  Returns 0.0 if either vector is zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------


def compute_drift(
    reference_texts: List[str],
    production_texts: List[str],
) -> Dict[str, Any]:
    """Compute distribution-level drift between two sets of texts.

    Builds a shared TF-IDF vocabulary, then measures:
    - **centroid_similarity**: cosine similarity between the mean TF-IDF
      vectors (1.0 = identical distributions, 0.0 = orthogonal).
    - **drift_score**: ``1 - centroid_similarity`` (higher = more drift).
    - **min_similarity**: the minimum per-production-prompt similarity to
      the reference centroid (identifies outlier prompts).

    Args:
        reference_texts: Texts from the reference distribution.
        production_texts: Texts from the production distribution.

    Returns:
        Dict with drift metrics and alert level.
    """
    all_texts = reference_texts + production_texts
    n_ref = len(reference_texts)

    matrix, vocab = build_tfidf_matrix(all_texts)
    ref_matrix = matrix[:n_ref]
    prod_matrix = matrix[n_ref:]

    # Centroid similarity
    ref_centroid = ref_matrix.mean(axis=0)
    prod_centroid = prod_matrix.mean(axis=0)
    centroid_sim = cosine_similarity(ref_centroid, prod_centroid)
    drift_score = 1.0 - centroid_sim

    # Per-prompt similarity to reference centroid
    per_prompt_sims = []
    for i in range(prod_matrix.shape[0]):
        sim = cosine_similarity(prod_matrix[i], ref_centroid)
        per_prompt_sims.append(float(sim))

    min_sim = min(per_prompt_sims) if per_prompt_sims else 0.0
    mean_sim = float(np.mean(per_prompt_sims)) if per_prompt_sims else 0.0

    level = alert_level(drift_score)

    return {
        "centroid_similarity": round(centroid_sim, 4),
        "drift_score": round(drift_score, 4),
        "min_prompt_similarity": round(min_sim, 4),
        "mean_prompt_similarity": round(mean_sim, 4),
        "alert_level": level,
        "num_reference": n_ref,
        "num_production": len(production_texts),
        "vocab_size": len(vocab),
    }


# ---------------------------------------------------------------------------
# Helpers: extract user prompts from chat JSONL
# ---------------------------------------------------------------------------


def extract_prompts(examples: List[Dict[str, Any]]) -> List[str]:
    """Extract user-role text from chat-format JSONL records.

    Each record is expected to have a ``messages`` key containing a list
    of ``{role, content}`` dicts.  Falls back to a ``prompt`` key if
    ``messages`` is absent.

    Args:
        examples: List of JSONL records.

    Returns:
        List of user prompt strings.
    """
    prompts: List[str] = []
    for ex in examples:
        if "messages" in ex:
            for msg in ex["messages"]:
                if msg.get("role") == "user":
                    prompts.append(msg["content"])
                    break
        elif "prompt" in ex:
            prompts.append(ex["prompt"])
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect prompt distribution drift between reference and production data"
    )
    parser.add_argument("--reference", required=True, help="Path to reference JSONL file")
    parser.add_argument("--production", required=True, help="Path to production JSONL file")
    parser.add_argument("--output", default="chapter09/eval/drift_report.json")
    args = parser.parse_args()

    # Load data
    ref_examples = list(read_jsonl(args.reference))
    prod_examples = list(read_jsonl(args.production))

    ref_prompts = extract_prompts(ref_examples)
    prod_prompts = extract_prompts(prod_examples)

    print(f"Reference prompts: {len(ref_prompts)}")
    print(f"Production prompts: {len(prod_prompts)}")

    # Compute drift
    report = compute_drift(ref_prompts, prod_prompts)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["reference_path"] = args.reference
    report["production_path"] = args.production

    # Print summary
    print(f"\nDrift score:         {report['drift_score']:.4f}")
    print(f"Centroid similarity: {report['centroid_similarity']:.4f}")
    print(f"Alert level:         {report['alert_level'].upper()}")

    if report["alert_level"] == "green":
        print("\nNo significant drift detected.")
    elif report["alert_level"] == "yellow":
        print("\nModerate drift detected -- investigate recent inputs.")
    else:
        print("\nHigh drift detected -- consider retraining or data refresh.")

    # Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
