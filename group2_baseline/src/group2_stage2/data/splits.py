from __future__ import annotations

import json
import random
from pathlib import Path

from ..common import load_jsonl, write_json, write_jsonl


def build_shared_quality_pool(
    stage2_root: Path,
    all_variants: list[str],
    quality_image_count: int,
    val_image_count: int,
    split_seed: int,
    pool_reference_variant: str,
    overwrite: bool = False,
) -> dict:
    variant_image_sets: dict[str, set] = {}
    for variant in all_variants:
        dataset_path = stage2_root / variant / "stage2_dataset.jsonl"
        rows = load_jsonl(dataset_path)
        variant_image_sets[variant] = {r["image_id"] for r in rows}

    common_image_ids = sorted(set.intersection(*(variant_image_sets[v] for v in all_variants)))
    if not common_image_ids:
        raise ValueError("No common images found across all variants.")

    effective_quality_count = min(quality_image_count, len(common_image_ids))
    if effective_quality_count < quality_image_count:
        print(
            "[build_shared_quality_pool] reducing quality_image_count "
            f"from {quality_image_count} to {effective_quality_count} "
            f"(common available: {len(common_image_ids)})"
        )

    rng = random.Random(split_seed)
    selected_image_ids = list(common_image_ids)
    rng.shuffle(selected_image_ids)
    selected_image_ids = selected_image_ids[:effective_quality_count]

    max_val_allowed = max(len(selected_image_ids) - 1, 0)
    effective_val_count = min(val_image_count, max_val_allowed)
    if effective_val_count < val_image_count:
        print(
            "[build_shared_quality_pool] reducing val_image_count "
            f"from {val_image_count} to {effective_val_count} "
            f"(selected quality: {len(selected_image_ids)})"
        )
    if effective_val_count <= 0:
        raise ValueError(
            "Not enough selected images to create a train/val split. "
            f"selected={len(selected_image_ids)} val={effective_val_count}"
        )

    val_image_ids = set(selected_image_ids[:effective_val_count])
    train_image_ids = set(selected_image_ids[effective_val_count:])

    pool_info = {
        "pool_reference_variant": pool_reference_variant,
        "all_variants": all_variants,
        "quality_image_count_requested": quality_image_count,
        "quality_image_count": effective_quality_count,
        "val_image_count_requested": val_image_count,
        "val_image_count": effective_val_count,
        "common_image_count_before_sampling": len(common_image_ids),
        "selected_image_ids": sorted(selected_image_ids),
    }
    split_info = {
        "seed": split_seed,
        "pool_reference_variant": pool_reference_variant,
        "all_variants": all_variants,
        "quality_image_count_requested": quality_image_count,
        "quality_image_count": effective_quality_count,
        "val_image_count_requested": val_image_count,
        "val_image_count": effective_val_count,
        "num_train_images": len(train_image_ids),
        "num_val_images": len(val_image_ids),
        "train_image_ids": sorted(train_image_ids),
        "val_image_ids": sorted(val_image_ids),
    }

    pool_write = write_json(stage2_root / "shared_quality_pool.json", pool_info, overwrite=overwrite)
    split_write = write_json(stage2_root / "shared_split.json", split_info, overwrite=overwrite)
    return {"pool_info": pool_info, "split_info": split_info, "pool_write": pool_write, "split_write": split_write}


def materialize_train_val_split(stage2_root: Path, all_variants: list[str], overwrite: bool = False) -> dict:
    pool_info = json.loads((stage2_root / "shared_quality_pool.json").read_text(encoding="utf-8"))
    split_info = json.loads((stage2_root / "shared_split.json").read_text(encoding="utf-8"))
    selected_ids = set(pool_info["selected_image_ids"])
    train_ids = set(split_info["train_image_ids"])
    val_ids = set(split_info["val_image_ids"])

    result: dict[str, dict] = {}
    for variant in all_variants:
        src = stage2_root / variant / "stage2_dataset.jsonl"
        rows = [r for r in load_jsonl(src) if r["image_id"] in selected_ids]
        train_rows = [r for r in rows if r["image_id"] in train_ids]
        val_rows = [r for r in rows if r["image_id"] in val_ids]
        train_write = write_jsonl(stage2_root / variant / "stage2_train.jsonl", train_rows, overwrite=overwrite)
        val_write = write_jsonl(stage2_root / variant / "stage2_val.jsonl", val_rows, overwrite=overwrite)
        result[variant] = {
            "pooled_rows": len(rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_write": train_write,
            "val_write": val_write,
        }
    return result
