from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict, overwrite: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(path)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"mode": "generated", "path": str(path)}


def write_jsonl(path: Path, rows: Iterable[dict], overwrite: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(path)}
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"mode": "generated", "path": str(path)}
