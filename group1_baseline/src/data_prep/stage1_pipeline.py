"""Stage-1 data pipeline orchestration helpers.

Keeps notebook cells thin by handling:
- load existing processed outputs when present
- regenerate only when requested
"""

from __future__ import annotations

import json
from pathlib import Path

from .prepare_stage1_dataset import build_alignment
from .convert_alignment_format import convert_alignment_rows


def ensure_stage1_chat_rows(
    coco_json_path: Path,
    alignment_path: Path,
    chat_path: Path,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[list[dict], list[dict], str]:
    """Return (alignment, chat_rows, mode) where mode is 'loaded' or 'generated'."""
    if not overwrite and alignment_path.exists() and chat_path.exists():
        with alignment_path.open("r", encoding="utf-8") as f:
            alignment = json.load(f)
        with chat_path.open("r", encoding="utf-8") as f:
            chat_rows = json.load(f)
        return alignment, chat_rows, "loaded"

    alignment = build_alignment(coco_json_path)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(json.dumps(alignment, indent=2), encoding="utf-8")

    chat_rows = convert_alignment_rows(alignment, seed=seed)
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    chat_path.write_text(json.dumps(chat_rows, indent=2), encoding="utf-8")

    return alignment, chat_rows, "generated"
