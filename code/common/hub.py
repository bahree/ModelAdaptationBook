"""Helpers for loading models/adapters from a Hugging Face repo subfolder.

The book's artifacts are published into a single unified repo
(``bahree/ModelAdaptationBook``) where each chapter's model or adapter lives in
its own subfolder (``ch5-lora``, ``ch6-sft``, ``ch7-distilled``, ``ch8-dpo``,
``ch8-dpo-lora``). To avoid adding a new CLI flag everywhere, any model/adapter
argument may use a ``repo#subfolder`` reference. ``split_ref`` turns that into
the ``(repo, subfolder)`` pair that ``from_pretrained`` expects, while leaving
plain local paths and plain HF repo ids untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def split_ref(ref: str) -> Tuple[str, Optional[str]]:
    """Split a model/adapter reference into (path_or_repo, subfolder_or_None).

    Lets any path/id argument also point into a Hugging Face repo subfolder
    via 'repo#subfolder', e.g. 'bahree/ModelAdaptationBook#ch6-sft'. A plain
    local path or HF repo id (no '#') returns (ref, None).
    """
    if ref and "#" in ref:
        repo, sub = ref.split("#", 1)
        return repo, (sub or None)
    return ref, None


def subfolder_kwargs(sub: Optional[str]) -> Dict[str, Any]:
    """Return the ``subfolder=`` kwarg for from_pretrained, or {} when None.

    transformers/peft ``from_pretrained`` joins the subfolder into a path with
    ``os.path.join(subfolder, ...)`` and crashes on ``subfolder=None`` (it
    expects ``""``). Splatting this helper passes the kwarg only when a real
    subfolder is present, so a plain local path or repo id is unaffected.
    """
    return {"subfolder": sub} if sub else {}
