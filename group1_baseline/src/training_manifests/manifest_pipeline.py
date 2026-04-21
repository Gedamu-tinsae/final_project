"""Manifest orchestration helpers for notebook-thin Stage 4."""

from __future__ import annotations

import json
from pathlib import Path

from .build_stage1_manifest import build_stage1_manifest
from .build_stage2_manifest import build_stage2_manifest


def _count_rows(json_path: Path) -> int:
    if not json_path.exists():
        return 0
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


def run_manifest_pipeline(
    *,
    stage1_tokenized_json: Path,
    stage2_tokenized_json: Path,
    clip_feature_dir: Path,
    stage1_manifest_json: Path,
    stage2_manifest_json: Path,
    overwrite: bool = False,
) -> dict:
    """Build Stage 1/2 manifests with skip behavior for existing outputs."""
    result = {
        "stage1_mode": "skipped_missing_input",
        "stage2_mode": "skipped_missing_input",
        "stage1_rows": 0,
        "stage2_rows": 0,
    }

    if not clip_feature_dir.exists():
        raise FileNotFoundError(f"Missing feature dir: {clip_feature_dir}")

    if stage1_tokenized_json.exists():
        if stage1_manifest_json.exists() and not overwrite:
            result["stage1_mode"] = "skipped_existing"
        else:
            stage1_manifest_json.parent.mkdir(parents=True, exist_ok=True)
            build_stage1_manifest(
                str(stage1_tokenized_json),
                str(clip_feature_dir),
                str(stage1_manifest_json),
                overwrite=overwrite,
            )
            result["stage1_mode"] = "generated"
        result["stage1_rows"] = _count_rows(stage1_manifest_json)

    if stage2_tokenized_json.exists():
        if stage2_manifest_json.exists() and not overwrite:
            result["stage2_mode"] = "skipped_existing"
        else:
            stage2_manifest_json.parent.mkdir(parents=True, exist_ok=True)
            build_stage2_manifest(
                str(stage2_tokenized_json),
                str(clip_feature_dir),
                str(stage2_manifest_json),
                overwrite=overwrite,
            )
            result["stage2_mode"] = "generated"
        result["stage2_rows"] = _count_rows(stage2_manifest_json)

    return result
