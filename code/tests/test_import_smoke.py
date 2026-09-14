"""Import-smoke test.

Imports every module that pulls the heavy ML dependency chain (peft, trl,
transformers) so CI catches dependency-resolution breaks on a fresh install --
the kind the chapter test suites miss because they only import lightweight data
and metrics modules.

This exists because a fresh install once resolved transformers to 5.x, which
removed ``HybridCache``; ``peft<0.18`` imports that at load time, so
``import peft`` (and the chapter 2 quickstart) failed everywhere -- yet CI
stayed green because no test imported peft. These imports are GPU-free and do
no training; they only exercise the import graph.

Deliberately excluded:
  - chapter05.train_qlora (imports bitsandbytes, a CUDA-only extra)
  - chapter03.ch03_data_quality_explore (a script with no __main__ guard, so
    importing it would run the full experiment)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make the code/ root importable when running pytest without an editable install.
_code_root = Path(__file__).resolve().parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

MODULES = [
    "chapter02.quickstart",
    "chapter05.modeling",
    "chapter05.train_lora",
    "chapter06.train_sft",
    "chapter07.train_student",
    "chapter08.train_dpo",
    "chapter09.safety_monitor",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    """Importing the module must not raise (catches dependency-resolution breaks).

    Chapters not present in this checkout are skipped, not failed: the public repo
    publishes chapters as the MEAP releases them, so a chapter's package may be absent.
    """
    if not (_code_root / module.split(".")[0]).is_dir():
        pytest.skip(f"{module.split('.')[0]} is not in this checkout")
    importlib.import_module(module)
