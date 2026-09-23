"""Unit tests for Chapter 6 evaluation helpers."""
from chapter06.eval_sft import extract_prompt_and_response


def test_extract_prompt_and_response_basic():
    example = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ],
        "category": "open_qa",
    }
    prompt, response, category = extract_prompt_and_response(example)
    assert prompt == "What is Python?"
    assert response == "A programming language."
    assert category == "open_qa"


def test_extract_prompt_and_response_no_system():
    example = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    }
    prompt, response, category = extract_prompt_and_response(example)
    assert prompt == "Hello"
    assert response == "Hi there"
    assert category == "unknown"


def test_extract_prompt_and_response_with_context():
    example = {
        "messages": [
            {"role": "system", "content": "You are an IT assistant."},
            {"role": "user", "content": "Linux is an OS.\n\nWhat is Linux?"},
            {"role": "assistant", "content": "Linux is an operating system."},
        ],
        "category": "closed_qa",
    }
    prompt, response, category = extract_prompt_and_response(example)
    assert "Linux" in prompt
    assert "operating system" in response
    assert category == "closed_qa"
