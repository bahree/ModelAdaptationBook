"""CLI wrapper for publishing a trained LoRA/QLoRA adapter.

Writes an adapter manifest (adapter_manifest.json), optionally exports the
adapter as a .tar.gz archive, and optionally pushes it to Hugging Face Hub.

Usage:
    python chapter05/scripts/publish_adapter.py \\
        --adapter chapter05/runs/it_lora \\
        --repo_id my-org/dolly-lora-adapter \\
        --private
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from chapter05 import DEFAULT_MODEL_NAME
from chapter05.publish import export_adapter_tarball, push_adapter_to_hub, write_adapter_manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for adapter publishing.

    Returns:
        Namespace with adapter path, base model info, Hub settings,
        and optional export/manifest options.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Adapter folder (output of training)")
    ap.add_argument("--base", default=DEFAULT_MODEL_NAME, help="Base model id")
    ap.add_argument("--base_revision", default=None, help="Pinned base model revision (commit hash or tag)")
    ap.add_argument("--tokenizer_revision", default=None, help="Pinned tokenizer revision (commit hash or tag)")
    ap.add_argument(
        "--system_prompt",
        default=None,
        help="System prompt used for training/eval (optional, for manifest bookkeeping).",
    )
    ap.add_argument(
        "--dataset_manifest",
        action="append",
        default=[],
        help="Path to a dataset manifest.json (repeatable).",
    )
    ap.add_argument(
        "--eval_report",
        default=None,
        help="Path to eval report JSON/MD (e.g. chapter05/runs/eval_report/report.json).",
    )
    ap.add_argument("--notes", default=None, help="Optional freeform notes to store in adapter_manifest.json")
    ap.add_argument("--repo_id", default=None, help="Hugging Face repo id (org/name) to push to")
    ap.add_argument("--private", action="store_true", help="Create a private repo on the Hub")
    ap.add_argument("--export_tar", default=None, help="If set, export adapter as a .tar.gz file to this path")
    ap.add_argument("--hf_token", default=None, help="HF token (or set HF_TOKEN env var)")
    return ap.parse_args()


def main() -> None:
    """Write manifest, optionally export tarball, and optionally push to Hub."""
    args = parse_args()
    adapter_dir = Path(args.adapter)

    token = args.hf_token or os.getenv("HF_TOKEN")

    manifest_path = write_adapter_manifest(
        adapter_dir,
        base_model=args.base,
        dataset_manifest_paths=list(args.dataset_manifest or []),
        eval_report_path=args.eval_report,
        base_model_revision=args.base_revision,
        tokenizer_revision=args.tokenizer_revision,
        system_prompt=args.system_prompt,
        notes=args.notes,
    )
    print(f"Wrote adapter manifest: {manifest_path}")

    if args.export_tar:
        tar_path = export_adapter_tarball(adapter_dir, args.export_tar)
        print(f"Exported adapter tarball: {tar_path}")

    if args.repo_id:
        if not token:
            # Fall back to the token cached by `huggingface-cli login`
            # at ~/.cache/huggingface/token.
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
        if not token:
            raise SystemExit(
                "HF token missing. Run `huggingface-cli login`, "
                "set HF_TOKEN env var, or pass --hf_token."
            )
        repo = push_adapter_to_hub(adapter_dir, repo_id=args.repo_id, private=args.private, token=token)
        print(f"Pushed adapter to Hub repo: {repo}")


if __name__ == "__main__":
    main()

