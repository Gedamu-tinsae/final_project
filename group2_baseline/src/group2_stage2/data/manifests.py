from __future__ import annotations

import json
from pathlib import Path

from .tokenization import resolve_stage2_paths


def _resolve_feature_path(image_name: str, feature_roots: list[Path]) -> Path | None:
    rel = Path(image_name).with_suffix(".npy")
    for root in feature_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def build_stage2_manifest(
    stage2_root: Path,
    feature_root: Path,
    variant: str,
    data_split: str = "train",
    overwrite: bool = False,
    additional_feature_roots: list[Path] | None = None,
) -> dict:
    variant_dir, _, tokenized_json = resolve_stage2_paths(stage2_root, variant, data_split)
    if data_split == "train":
        manifest_json = variant_dir / "stage2_manifest_train.json"
    elif data_split == "val":
        manifest_json = variant_dir / "stage2_manifest_val.json"
    else:
        manifest_json = variant_dir / "stage2_manifest_full.json"

    if manifest_json.exists() and not overwrite:
        return {
            "mode": "skipped_existing",
            "variant": variant,
            "split": data_split,
            "manifest_json": str(manifest_json),
        }

    stage2_data = json.loads(tokenized_json.read_text(encoding="utf-8"))
    feature_roots = [feature_root] + list(additional_feature_roots or [])
    manifest = []
    missing_features = 0
    reused_features = 0
    for row in stage2_data:
        vision_path = _resolve_feature_path(row["image"], feature_roots)
        if vision_path is not None:
            primary_path = (feature_root / Path(row["image"])).with_suffix(".npy")
            if vision_path != primary_path:
                reused_features += 1
            manifest.append(
                {
                    "vision_path": str(vision_path),
                    "input_ids": row["input_ids"],
                    "labels": row["labels"],
                    "image": row.get("image"),
                    "image_id": row.get("image_id"),
                    "task_type": row.get("task_type", "unknown"),
                    "generator_model": row.get("generator_model"),
                    "sample_id": row.get("sample_id"),
                    "variant": variant,
                    "data_split": data_split,
                }
            )
        else:
            missing_features += 1
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return {
        "mode": "generated",
        "variant": variant,
        "split": data_split,
        "rows": len(manifest),
        "missing_features": missing_features,
        "reused_features": reused_features,
        "manifest_json": str(manifest_json),
    }
