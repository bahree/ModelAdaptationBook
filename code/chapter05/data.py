"""Chat data loading and normalization for Chapter 5.

- ChatExample: one example with messages (system/user/assistant).
- load_chat_jsonl(): load JSONL (messages or prompt/response) into ChatExamples.
- normalize_row_to_chat_example(): normalize a JSONL row to ChatExample.
Used by training scripts and eval to read Dolly-style or messages JSONL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from common.jsonl import read_jsonl


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatExample:
    messages: List[Dict[str, str]]


def _validate_messages(messages: Any, *, context: str) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{context}: 'messages' must be a non-empty list")
    out: List[Dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"{context}: message[{i}] must be an object")
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{context}: message[{i}].role must be system/user/assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{context}: message[{i}].content must be a non-empty string")
        out.append({"role": role, "content": content})
    return out


def prompt_response_to_messages(
    prompt: str,
    response: str,
    *,
    system_prompt: str,
) -> List[Dict[str, str]]:
    # Treat the provided prompt as the user message. This is intentionally simple and
    # keeps compatibility with early drafts that used prompt/response JSONL.
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def normalize_row_to_chat_example(
    row: Dict[str, Any],
    *,
    system_prompt: str,
    context: str,
) -> ChatExample:
    if "messages" in row:
        messages = _validate_messages(row["messages"], context=context)
        # Ensure a system prompt exists (Qwen chat models behave best with one).
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
        return ChatExample(messages=messages)

    if "prompt" in row and "response" in row:
        prompt = row["prompt"]
        response = row["response"]
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"{context}: 'prompt' and 'response' must be strings")
        messages = prompt_response_to_messages(prompt, response, system_prompt=system_prompt)
        return ChatExample(messages=messages)

    raise ValueError(
        f"{context}: expected either {{messages:[...]}} or {{prompt:..., response:...}}"
    )


def load_chat_jsonl(
    path: str | Path,
    *,
    system_prompt: str,
) -> List[ChatExample]:
    p = Path(path)
    out: List[ChatExample] = []
    for idx, row in enumerate(read_jsonl(p), start=1):
        out.append(
            normalize_row_to_chat_example(
                row,
                system_prompt=system_prompt,
                context=f"{p.name}:{idx}",
            )
        )
    if not out:
        raise ValueError(f"{p} contained no examples")
    return out

