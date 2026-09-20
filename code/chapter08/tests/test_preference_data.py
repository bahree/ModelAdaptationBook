"""Tests for Chapter 8 preference data format and integrity."""
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "preference_pairs"


@pytest.fixture
def train_data():
    path = DATA_DIR / "train.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def valid_data():
    path = DATA_DIR / "valid.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def manifest():
    path = DATA_DIR / "manifest.json"
    with open(path) as f:
        return json.load(f)


def test_train_valid_counts(train_data, valid_data, manifest):
    """Train/valid counts match manifest."""
    assert len(train_data) == manifest["counts"]["train"]
    assert len(valid_data) == manifest["counts"]["valid"]


def test_preference_pair_schema(train_data):
    """Each record has prompt, chosen, and rejected fields."""
    for i, record in enumerate(train_data):
        assert "prompt" in record, f"Record {i} missing 'prompt'"
        assert "chosen" in record, f"Record {i} missing 'chosen'"
        assert "rejected" in record, f"Record {i} missing 'rejected'"


def test_prompt_is_message_list(train_data):
    """Prompt field is a list of chat messages with system + user roles."""
    for i, record in enumerate(train_data):
        prompt = record["prompt"]
        assert isinstance(prompt, list), f"Record {i}: prompt is not a list"
        assert len(prompt) >= 2, f"Record {i}: prompt has < 2 messages"
        roles = [m["role"] for m in prompt]
        assert "system" in roles, f"Record {i}: no system message"
        assert "user" in roles, f"Record {i}: no user message"


def test_chosen_rejected_are_nonempty(train_data):
    """Chosen and rejected responses contain non-empty content."""
    for i, record in enumerate(train_data):
        chosen = record["chosen"]
        rejected = record["rejected"]
        # Handle both list and dict formats
        if isinstance(chosen, list):
            content = chosen[0].get("content", "")
        else:
            content = chosen.get("content", "")
        assert len(content.strip()) > 0, f"Record {i}: chosen is empty"

        if isinstance(rejected, list):
            content = rejected[0].get("content", "")
        else:
            content = rejected.get("content", "")
        assert len(content.strip()) > 0, f"Record {i}: rejected is empty"


def test_manifest_has_source_models(manifest):
    """Manifest records which models generated chosen/rejected."""
    assert "chosen_model" in manifest
    assert "rejected_model" in manifest
    assert manifest["seed"] == 42
