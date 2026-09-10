from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the code/ directory (one level up from common/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)


def _guard_hf_transfer() -> None:
    """Keep Hugging Face downloads working when HF_HUB_ENABLE_HF_TRANSFER is set
    (some cloud images and notebooks export it) but the `hf_transfer` package is
    not installed: huggingface_hub then refuses to download at all. The book's
    code never needs hf_transfer, so fall back to the standard downloader."""
    flag = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        import sys
        consts = sys.modules.get("huggingface_hub.constants")
        if consts is not None:
            consts.HF_HUB_ENABLE_HF_TRANSFER = False
        print("[common.env] HF_HUB_ENABLE_HF_TRANSFER was set but hf_transfer is not installed; "
              "using the standard downloader (pip install hf_transfer to re-enable).")


_guard_hf_transfer()


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    return v


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def resolve_report_to(cli_report_to: str | None) -> str:
    """Resolve `report_to` with safe defaults.

    Order of precedence:
    - CLI flag if present
    - BOOKCODE_REPORT_TO env var if present
    - default: "none"
    - WANDB_DISABLED=true forces "none"
    """
    if env_bool("WANDB_DISABLED", default=False):
        return "none"

    if cli_report_to:
        return cli_report_to

    return env_str("BOOKCODE_REPORT_TO", default="none") or "none"
