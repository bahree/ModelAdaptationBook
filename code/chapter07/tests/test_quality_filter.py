"""Tests for the distillation data quality filter."""
from chapter07.scripts.prepare_distillation_data import quality_filter


def test_accepts_normal_response():
    ex = {
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, go to Settings, "
             "click Security, then click Reset Password. Follow the on-screen instructions."},
        ]
    }
    assert quality_filter(ex, min_words=10, max_words=500) is True


def test_rejects_too_short():
    ex = {
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "How do I fix this?"},
            {"role": "assistant", "content": "Try restarting."},
        ]
    }
    assert quality_filter(ex, min_words=10, max_words=500) is False


def test_rejects_too_long():
    ex = {
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Explain everything."},
            {"role": "assistant", "content": " ".join(["word"] * 600)},
        ]
    }
    assert quality_filter(ex, min_words=10, max_words=500) is False


def test_rejects_degenerate_repetition():
    repeated = "This is a test sentence. " * 20
    ex = {
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": repeated},
        ]
    }
    assert quality_filter(ex, min_words=10, max_words=500) is False
