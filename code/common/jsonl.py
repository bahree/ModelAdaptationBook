from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {p}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {p}")
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_list(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def require_keys(obj: Dict[str, Any], keys: Iterable[str], *, context: str = "") -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}missing required keys: {missing}")


def as_str(obj: Dict[str, Any], key: str, *, context: str = "") -> str:
    v = obj.get(key)
    if not isinstance(v, str):
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}expected '{key}' to be a string")
    return v


def as_list(obj: Dict[str, Any], key: str, *, context: str = "") -> List[Any]:
    v = obj.get(key)
    if not isinstance(v, list):
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}expected '{key}' to be a list")
    return v
