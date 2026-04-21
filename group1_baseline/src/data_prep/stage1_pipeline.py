"""Stage-1 data pipeline orchestration helpers.

Keeps notebook cells thin by handling:
- load existing processed outputs when present
- regenerate only when requested
"""

from __future__ import annotations

import json
from pathlib import Path

from .acquire_coco import acquire_coco_2017, coco_files_status
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


def run_stage1_data_prep(
    project_root: Path,
    coco_json_path: Path,
    alignment_path: Path,
    chat_path: Path,
    *,
    seed: int = 42,
    overwrite: bool = False,
    download: bool = True,
    extract: bool = True,
) -> tuple[list[dict], list[dict], str, dict[str, Path]]:
    """Run full Stage-1 prep: acquisition/status + alignment/chat generation.

    Returns:
      alignment, chat_rows, mode, status
    """
    acquire_coco_2017(project_root, download=download, extract=extract)
    status = coco_files_status(project_root)
    alignment, chat_rows, mode = ensure_stage1_chat_rows(
        coco_json_path=coco_json_path,
        alignment_path=alignment_path,
        chat_path=chat_path,
        seed=seed,
        overwrite=overwrite,
    )
    return alignment, chat_rows, mode, status
