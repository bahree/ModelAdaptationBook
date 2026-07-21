"""Adapter publishing utilities: validation, manifest writing, tarball export, and Hub push.

Provides the building blocks for packaging a trained LoRA/QLoRA adapter for
distribution. The typical workflow is:
    1. ``validate_adapter_dir`` - check that the folder contains valid adapter files.
    2. ``write_adapter_manifest`` - record metadata (base model, datasets, eval results).
    3. ``export_adapter_tarball`` - create a portable .tar.gz archive.
    4. ``push_adapter_to_hub`` - upload to Hugging Face Hub.

Used by ``scripts/publish_adapter.py``.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import HfApi

from common.manifest import write_json


def validate_adapter_dir(adapter_dir: str | Path) -> Path:
    """Validate that a directory contains a PEFT adapter (config + weights).

    Args:
        adapter_dir: Path to the adapter directory.

    Returns:
        Resolved Path to the adapter directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If adapter_config.json or weight files are missing.
    """
    p = Path(adapter_dir)
    if not p.exists():
        raise FileNotFoundError(p)
    cfg = p / "adapter_config.json"
    if not cfg.exists():
        raise ValueError(f"Not a PEFT adapter folder (missing {cfg.name}): {p}")
    # Weights can be safetensors or bin format depending on the PEFT version.
    weights = list(p.glob("adapter_model.*"))
    if not weights:
        # PEFT may also name weights differently; keep it permissive.
        weights = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    if not weights:
        raise ValueError(f"Adapter folder has no weights: {p}")
    return p


def write_adapter_manifest(
    adapter_dir: str | Path,
    *,
    base_model: str,
    dataset_manifest_paths: list[str] | None = None,
    eval_report_path: str | None = None,
    base_model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    system_prompt: str | None = None,
    notes: str | None = None,
) -> Path:
    """Write an adapter_manifest.json with metadata about the adapter.

    The manifest records provenance information (base model, dataset, eval
    results) so that anyone receiving the adapter can reproduce or audit it.

    Args:
        adapter_dir: Path to the adapter directory.
        base_model: HuggingFace model ID for the base model.
        dataset_manifest_paths: Paths to dataset manifest.json files.
        eval_report_path: Path to the evaluation report JSON.
        base_model_revision: Git commit hash or tag for the base model.
        tokenizer_revision: Git commit hash or tag for the tokenizer.
        system_prompt: System prompt used during training/eval.
        notes: Optional freeform notes.

    Returns:
        Path to the written adapter_manifest.json.
    """
    p = validate_adapter_dir(adapter_dir)
    manifest: Dict[str, Any] = {
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "system_prompt": system_prompt,
        "adapter_dir": str(p),
        "dataset_manifests": dataset_manifest_paths or [],
        "eval_report": eval_report_path or None,
        "notes": notes,
    }
    out = p / "adapter_manifest.json"
    write_json(out, manifest)
    return out


def export_adapter_tarball(adapter_dir: str | Path, out_path: str | Path) -> Path:
    """Export an adapter directory as a .tar.gz archive.

    Includes all files in the adapter directory (config, weights, manifest,
    tokenizer files). Useful for sharing adapters without Hub access.

    Args:
        adapter_dir: Path to the validated adapter directory.
        out_path: Destination path for the .tar.gz file.

    Returns:
        Path to the created tarball.
    """
    p = validate_adapter_dir(adapter_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        for f in p.rglob("*"):
            if f.is_file():
                tf.add(f, arcname=f.relative_to(p))
    return out


def push_adapter_to_hub(
    adapter_dir: str | Path,
    *,
    repo_id: str,
    private: bool,
    token: str | None = None,
) -> str:
    """Upload an adapter directory to Hugging Face Hub.

    Creates the repo if it doesn't exist, then uploads all adapter files.

    Args:
        adapter_dir: Path to the validated adapter directory.
        repo_id: Hub repository ID (e.g., "my-org/dolly-lora-adapter").
        private: Whether to create a private repository.
        token: Hugging Face API token (or set HF_TOKEN env var).

    Returns:
        The repo_id of the uploaded adapter.
    """
    p = validate_adapter_dir(adapter_dir)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(p),
        commit_message="Add Chapter 5 adapter",
    )
    return repo_id
