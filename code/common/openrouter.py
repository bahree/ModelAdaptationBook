"""Thin OpenRouter helper.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint that
fronts Anthropic, Google, OpenAI, DeepSeek, and others under a single
key. We use it to capture a frontier-model response next to a local
SFT model's response without depending on multiple vendor SDKs.

The key is read from ``OPENROUTER_API_KEY`` (loaded from ``code/.env``
by ``common.env``).

Returned dict shape::

    {
        "content":   str,                 # user-facing answer
        "reasoning": str | None,          # thinking-mode preamble (Gemini, etc.)
        "usage":     {...},               # OpenRouter-reported token usage
        "model":     str,                 # echoed model id
        "raw":       dict,                # full JSON response (for audit)
    }
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

# Importing common.env triggers load_dotenv() of code/.env at import time.
import common.env  # noqa: F401  (side-effect import: loads OPENROUTER_API_KEY)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 60
# Required by OpenRouter for attribution headers; used in usage analytics.
DEFAULT_REFERER = "https://github.com/bahree/ModelAdaptationBook"
DEFAULT_TITLE = "Practical Model Adaptation Techniques (book)"


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns a non-200 response."""


def chat(
    messages: List[Dict[str, str]],
    model: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    extra_body: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Call OpenRouter's chat-completions endpoint and return the parsed result.

    Args:
        messages: Chat messages in OpenAI format ([{"role": "user", "content": "..."}]).
        model: OpenRouter model id (e.g. ``anthropic/claude-sonnet-5.5``).
        max_tokens: Cap on total generated tokens. Reasoning-mode models (Gemini)
            consume some of this budget on hidden reasoning before emitting
            ``content``; default 1024 is enough for typical chapter prompts.
        temperature: Sampling temperature. Default 0 for reproducible book examples.
        timeout: Per-call HTTP timeout in seconds.
        extra_body: Any additional fields to merge into the request body
            (e.g. ``{"top_p": 0.9}``).

    Raises:
        OpenRouterError: HTTP non-200 or missing ``content`` field in the response.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Add it to code/.env or your shell env."
        )

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": DEFAULT_REFERER,
        "X-Title": DEFAULT_TITLE,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
    if response.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter HTTP {response.status_code} for model={model}: {response.text[:500]}"
        )

    data = response.json()
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"Unexpected response shape for {model}: {data}") from exc

    content = msg.get("content")
    if content is None:
        raise OpenRouterError(
            f"Model {model} returned no content. Likely max_tokens was consumed by "
            f"reasoning preamble; raise max_tokens. Raw: {data}"
        )

    return {
        "content": content,
        "reasoning": msg.get("reasoning"),
        "usage": data.get("usage", {}),
        "model": data.get("model", model),
        "raw": data,
    }
