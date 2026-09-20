"""Unit tests for the drift detector (Listing 9.2)."""
from __future__ import annotations

from chapter09.drift_detector import (
    alert_level,
    build_tfidf_matrix,
    compute_drift,
    cosine_similarity,
    extract_prompts,
)
import numpy as np


def test_no_drift_identical_data():
    """Identical reference and production data should produce near-zero drift."""
    texts = [
        "How do I reset my password?",
        "What is the VPN setup process?",
        "My laptop won't connect to WiFi.",
        "How do I install the printer driver?",
        "Where can I find the employee handbook?",
    ]
    report = compute_drift(texts, texts)
    assert report["drift_score"] < 0.05
    assert report["alert_level"] == "green"


def test_drift_detected_different_data():
    """Very different data should produce high drift."""
    reference = [
        "How do I reset my password?",
        "What is the VPN setup process?",
        "My laptop won't connect to WiFi.",
        "How do I install the printer driver?",
        "Where can I find the employee handbook?",
    ]
    production = [
        "What is the best recipe for chocolate cake?",
        "How do I train for a marathon?",
        "Explain quantum entanglement.",
        "What are the best vacation spots in Europe?",
        "How do I learn to play guitar?",
    ]
    report = compute_drift(reference, production)
    assert report["drift_score"] > 0.15
    assert report["alert_level"] in ("yellow", "red")


def test_alert_levels():
    """Verify the green / yellow / red thresholds."""
    assert alert_level(0.0) == "green"
    assert alert_level(0.05) == "green"
    assert alert_level(0.09) == "green"
    assert alert_level(0.10) == "yellow"
    assert alert_level(0.15) == "yellow"
    assert alert_level(0.19) == "yellow"
    assert alert_level(0.20) == "red"
    assert alert_level(0.50) == "red"
    assert alert_level(1.0) == "red"


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    sim = cosine_similarity(a, a)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sim = cosine_similarity(a, b)
    assert abs(sim) < 1e-6


def test_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_build_tfidf_matrix_shape():
    docs = ["hello world", "foo bar baz", "hello foo"]
    matrix, vocab = build_tfidf_matrix(docs)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vocab)
    assert len(vocab) > 0


def test_extract_prompts_messages_format():
    """Extract prompts from chat-format JSONL records."""
    examples = [
        {"messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to settings."},
        ]},
        {"messages": [
            {"role": "user", "content": "What is VPN?"},
            {"role": "assistant", "content": "A virtual private network."},
        ]},
    ]
    prompts = extract_prompts(examples)
    assert len(prompts) == 2
    assert prompts[0] == "How do I reset my password?"
    assert prompts[1] == "What is VPN?"


def test_extract_prompts_prompt_format():
    """Extract prompts from simple prompt-key records."""
    examples = [
        {"prompt": "How do I reset my password?"},
        {"prompt": "What is VPN?"},
    ]
    prompts = extract_prompts(examples)
    assert len(prompts) == 2


def test_drift_report_has_required_keys():
    """Verify the drift report contains all expected keys."""
    texts = ["hello world", "foo bar"]
    report = compute_drift(texts, texts)
    expected_keys = {
        "centroid_similarity", "drift_score", "min_prompt_similarity",
        "mean_prompt_similarity", "alert_level", "num_reference",
        "num_production", "vocab_size",
    }
    assert expected_keys.issubset(report.keys())
