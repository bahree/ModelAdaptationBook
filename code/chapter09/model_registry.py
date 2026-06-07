"""Listing 9.1 -- JSON-based model registry for tracking model versions.

A lightweight model registry that tracks model versions, promotions, and
rollbacks without requiring MLflow or any external service.  Each entry
records the technique (lora/sft/distill/dpo), base model, data hash,
hyperparameters, evaluation metrics, and lifecycle status.

The registry is a single JSON file -- easy to inspect, diff, and commit
to version control alongside the code that produced each model.

Run from code/:
    python -m chapter09.model_registry register \
        --name it-support-v2 --technique lora \
        --base_model Qwen/Qwen3-4B-Instruct-2507 \
        --data_hash abc123 --checkpoint_path chapter05/runs/lora_r8 \
        --eval_metrics '{"overall_f1": 0.72}' \
        --registry_dir chapter09/data

    python -m chapter09.model_registry list --registry_dir chapter09/data
    python -m chapter09.model_registry promote --version_tag it-support-v2 --registry_dir chapter09/data
    python -m chapter09.model_registry rollback --registry_dir chapter09/data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_FILENAME = "registry.json"


def _load_registry(registry_dir: str | Path) -> List[Dict[str, Any]]:
    """Load the registry from disk, returning an empty list if absent."""
    path = Path(registry_dir) / REGISTRY_FILENAME
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(registry_dir: str | Path, entries: List[Dict[str, Any]]) -> None:
    """Persist the registry to disk."""
    path = Path(registry_dir) / REGISTRY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_version_tag(
    name: str,
    version_num: int,
    technique: str,
    base_model: str,
    data_version: str = "v1",
) -> str:
    """Build a human-readable version tag.

    Format: ``{name}-v{N}-{date}-{technique}-{base_short}-{data_version}``

    Args:
        name: Model family name (e.g. ``it-support``).
        version_num: Sequential version number.
        technique: Training technique (lora, sft, distill, dpo).
        base_model: Full base-model identifier (e.g. ``Qwen/Qwen3-4B-Instruct-2507``).
        data_version: Data version label (default ``v1``).

    Returns:
        A version tag string.
    """
    date_str = time.strftime("%Y%m%d")
    base_short = base_model.split("/")[-1][:20]
    return f"{name}-v{version_num}-{date_str}-{technique}-{base_short}-{data_version}"


def register_model(
    registry_dir: str | Path,
    version_tag: str,
    technique: str,
    base_model: str,
    data_hash: str,
    checkpoint_path: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Register a new model version in the registry.

    Args:
        registry_dir: Directory containing ``registry.json``.
        version_tag: Unique version identifier.
        technique: Training technique (lora/sft/distill/dpo).
        base_model: Base model identifier.
        data_hash: Hash or identifier of the training data.
        checkpoint_path: Path to the model checkpoint / adapter.
        hyperparameters: Training hyperparameters dict (optional).
        eval_metrics: Evaluation metrics dict (optional).
        notes: Free-text notes (optional).

    Returns:
        The newly created registry entry.

    Raises:
        ValueError: If a version with the same tag already exists.
    """
    entries = _load_registry(registry_dir)

    # Check for duplicate version tag
    existing_tags = {e["version_tag"] for e in entries}
    if version_tag in existing_tags:
        raise ValueError(f"Version tag '{version_tag}' already exists in registry")

    entry = {
        "version_tag": version_tag,
        "technique": technique,
        "base_model": base_model,
        "data_hash": data_hash,
        "hyperparameters": hyperparameters or {},
        "eval_metrics": eval_metrics or {},
        "checkpoint_path": checkpoint_path,
        "status": "registered",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes,
    }
    entries.append(entry)
    _save_registry(registry_dir, entries)
    return entry


def list_versions(registry_dir: str | Path) -> List[Dict[str, Any]]:
    """Return all registered model versions."""
    return _load_registry(registry_dir)


def get_version(registry_dir: str | Path, version_tag: str) -> Optional[Dict[str, Any]]:
    """Look up a specific version by tag.

    Returns:
        The matching entry, or ``None`` if not found.
    """
    for entry in _load_registry(registry_dir):
        if entry["version_tag"] == version_tag:
            return entry
    return None


def get_active_version(registry_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Return the currently active (deployed) version, if any."""
    for entry in _load_registry(registry_dir):
        if entry["status"] == "active":
            return entry
    return None


def get_latest_version(registry_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Return the most recently registered version."""
    entries = _load_registry(registry_dir)
    return entries[-1] if entries else None


def promote_version(registry_dir: str | Path, version_tag: str) -> Dict[str, Any]:
    """Promote a version to active, retiring the currently active version.

    Args:
        registry_dir: Directory containing ``registry.json``.
        version_tag: Version tag to promote.

    Returns:
        The promoted entry.

    Raises:
        ValueError: If the version tag is not found.
    """
    entries = _load_registry(registry_dir)
    target = None
    for entry in entries:
        if entry["version_tag"] == version_tag:
            target = entry
        elif entry["status"] == "active":
            entry["status"] = "retired"
    if target is None:
        raise ValueError(f"Version tag '{version_tag}' not found in registry")
    target["status"] = "active"
    target["promoted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_registry(registry_dir, entries)
    return target


def rollback(registry_dir: str | Path) -> Dict[str, Any]:
    """Roll back to the previously active version.

    Finds the most recent *retired* version and promotes it back to
    active, retiring the currently active version.

    Returns:
        The re-activated entry.

    Raises:
        ValueError: If no retired version is available for rollback.
    """
    entries = _load_registry(registry_dir)

    # Find the most recent retired version (candidate for rollback)
    retired = [e for e in entries if e["status"] == "retired"]
    if not retired:
        raise ValueError("No retired version available for rollback")
    rollback_target = retired[-1]

    # Retire current active
    for entry in entries:
        if entry["status"] == "active":
            entry["status"] = "retired"

    rollback_target["status"] = "active"
    rollback_target["promoted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_registry(registry_dir, entries)
    return rollback_target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_register(args: argparse.Namespace) -> None:
    eval_metrics = json.loads(args.eval_metrics) if args.eval_metrics else {}
    hyperparams = json.loads(args.hyperparameters) if args.hyperparameters else {}
    entry = register_model(
        registry_dir=args.registry_dir,
        version_tag=args.name,
        technique=args.technique,
        base_model=args.base_model,
        data_hash=args.data_hash,
        checkpoint_path=args.checkpoint_path,
        hyperparameters=hyperparams,
        eval_metrics=eval_metrics,
        notes=args.notes,
    )
    print(f"Registered: {entry['version_tag']}  status={entry['status']}")


def _cli_list(args: argparse.Namespace) -> None:
    entries = list_versions(args.registry_dir)
    if not entries:
        print("Registry is empty.")
        return
    print(f"{'Version Tag':<55} {'Status':<12} {'Technique':<10} {'Created'}")
    print("-" * 100)
    for e in entries:
        print(f"{e['version_tag']:<55} {e['status']:<12} {e['technique']:<10} {e['created_utc']}")


def _cli_promote(args: argparse.Namespace) -> None:
    entry = promote_version(args.registry_dir, args.version_tag)
    print(f"Promoted: {entry['version_tag']} -> active")


def _cli_rollback(args: argparse.Namespace) -> None:
    entry = rollback(args.registry_dir)
    print(f"Rolled back to: {entry['version_tag']} -> active")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model registry: register, list, promote, and rollback model versions"
    )
    parser.add_argument(
        "--registry_dir", default="chapter09/data", help="Directory for registry.json"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # register
    reg = sub.add_parser("register", help="Register a new model version")
    reg.add_argument("--name", required=True, help="Version tag / model name")
    reg.add_argument("--technique", required=True, choices=["lora", "sft", "distill", "dpo"])
    reg.add_argument("--base_model", required=True, help="Base model identifier")
    reg.add_argument("--data_hash", required=True, help="Hash or ID of training data")
    reg.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    reg.add_argument("--eval_metrics", default=None, help="JSON string of eval metrics")
    reg.add_argument("--hyperparameters", default=None, help="JSON string of hyperparameters")
    reg.add_argument("--notes", default="", help="Free-text notes")
    reg.set_defaults(func=_cli_register)

    # list
    ls = sub.add_parser("list", help="List all registered versions")
    ls.set_defaults(func=_cli_list)

    # promote
    prom = sub.add_parser("promote", help="Promote a version to active")
    prom.add_argument("--version_tag", required=True, help="Version tag to promote")
    prom.set_defaults(func=_cli_promote)

    # rollback
    rb = sub.add_parser("rollback", help="Rollback to previous active version")
    rb.set_defaults(func=_cli_rollback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
