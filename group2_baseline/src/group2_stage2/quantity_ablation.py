from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Callable

from .common import load_jsonl, write_jsonl


def derive_quantity_plan(stage2_root: Path, quality_image_count: int | None = None, val_image_count: int | None = None) -> dict:
    engine_summary = json.loads((stage2_root / "engine_comparison_summary.json").read_text(encoding="utf-8"))
    ranking = engine_summary.get("variant_summaries", [])
    if not ranking:
        raise ValueError("engine_comparison_summary.json has no variant_summaries entries.")
    quantity_source_variant = engine_summary.get("best_variant") or ranking[0]["variant"]

    if quality_image_count is None:
        pool_info = json.loads((stage2_root / "shared_quality_pool.json").read_text(encoding="utf-8"))
        quality_image_count = int(pool_info.get("quality_image_count", 5000))
    if val_image_count is None:
        split_info = json.loads((stage2_root / "shared_split.json").read_text(encoding="utf-8"))
        val_image_count = int(split_info.get("num_val_images", 1000))

    quantity_levels = [quality_image_count - 2000, quality_image_count - 1000, quality_image_count]
    if min(quantity_levels) <= 0:
        raise ValueError(f"Invalid quantity levels: {quantity_levels}")
    return {
        "quantity_source_variant": quantity_source_variant,
        "quantity_base_total_images": quality_image_count,
        "quantity_val_image_count": val_image_count,
        "quantity_levels": quantity_levels,
        "quantity_split_seed": 123,
    }


def build_quantity_variants(stage2_root: Path, quantity_source_variant: str, quantity_levels: list[int], quantity_split_seed: int) -> list[str]:
    pool_info = json.loads((stage2_root / "shared_quality_pool.json").read_text(encoding="utf-8"))
    split_info = json.loads((stage2_root / "shared_split.json").read_text(encoding="utf-8"))
    selected_ids = set(pool_info["selected_image_ids"])
    base_train_ids = sorted(split_info["train_image_ids"])
    base_val_ids = sorted(split_info["val_image_ids"])

    num_base_train = len(base_train_ids)
    num_base_val = len(base_val_ids)
    source_dataset = stage2_root / quantity_source_variant / "stage2_dataset.jsonl"
    rng = random.Random(quantity_split_seed)
    shuffled_train_ids = base_train_ids[:]
    rng.shuffle(shuffled_train_ids)

    quantity_root = stage2_root / "quantity_ablation"
    quantity_root.mkdir(parents=True, exist_ok=True)
    quantity_variants: list[str] = []

    rows_all = [r for r in load_jsonl(source_dataset) if r["image_id"] in selected_ids]
    for qty in quantity_levels:
        qty_train_count = qty - num_base_val
        if qty_train_count <= 0:
            raise ValueError(f"Requested total quantity {qty} but fixed validation size is {num_base_val}.")
        if qty_train_count > num_base_train:
            raise ValueError(f"Requested total quantity {qty} needs {qty_train_count} train images; only {num_base_train} available.")

        qty_variant = f"{quantity_source_variant}_qty_{qty}"
        qty_dir = quantity_root / qty_variant
        qty_dir.mkdir(parents=True, exist_ok=True)
        qty_train_ids = set(shuffled_train_ids[:qty_train_count])
        qty_val_ids = set(base_val_ids)

        train_rows = [r for r in rows_all if r["image_id"] in qty_train_ids]
        val_rows = [r for r in rows_all if r["image_id"] in qty_val_ids]
        write_jsonl(qty_dir / "stage2_train.jsonl", train_rows, overwrite=True)
        write_jsonl(qty_dir / "stage2_val.jsonl", val_rows, overwrite=True)

        metadata = {
            "source_variant": quantity_source_variant,
            "quantity_level_total_images": qty,
            "num_train_images": len(qty_train_ids),
            "num_val_images": len(qty_val_ids),
            "num_total_images": len(qty_train_ids) + len(qty_val_ids),
            "train_image_ids": sorted(qty_train_ids),
            "val_image_ids": sorted(qty_val_ids),
            "split_seed": quantity_split_seed,
        }
        (qty_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        quantity_variants.append(qty_variant)
    return quantity_variants


def register_quantity_variants(stage2_root: Path, quantity_variants: list[str], overwrite: bool = False) -> dict:
    registration_path = stage2_root / "quantity_registration_status.json"
    try:
        registration_status = json.loads(registration_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        registration_status = {}

    required_files = ["stage2_train.jsonl", "stage2_val.jsonl", "metadata.json"]
    for qty_variant in quantity_variants:
        src_dir = stage2_root / "quantity_ablation" / qty_variant
        dst_dir = stage2_root / qty_variant
        dst_dir.mkdir(parents=True, exist_ok=True)
        for filename in required_files:
            dst_file = dst_dir / filename
            if dst_file.exists() and not overwrite:
                continue
            shutil.copy2(src_dir / filename, dst_file)
        registration_status[qty_variant] = {
            "status": "registered",
            "source_dir": str(src_dir),
            "dest_dir": str(dst_dir),
            "files": [str(dst_dir / f) for f in required_files],
        }
    registration_path.write_text(json.dumps(registration_status, ensure_ascii=False, indent=2), encoding="utf-8")
    return registration_status


def prepare_quantity_variants(
    stage2_root: Path,
    quantity_variants: list[str],
    tokenize_fn: Callable[[str, str], dict],
    extract_features_fn: Callable[[str, str], dict],
    build_manifest_fn: Callable[[str, str], dict],
    overwrite: bool = False,
) -> dict:
    prep_status_path = stage2_root / "quantity_prep_status.json"
    try:
        prep_status = json.loads(prep_status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        prep_status = {}

    for variant in quantity_variants:
        variant_root = stage2_root / variant
        train_manifest = variant_root / "stage2_manifest_train.json"
        val_manifest = variant_root / "stage2_manifest_val.json"
        if train_manifest.exists() and val_manifest.exists() and not overwrite:
            prep_status[variant] = {"status": "ready", "train_manifest": str(train_manifest), "val_manifest": str(val_manifest)}
            prep_status_path.write_text(json.dumps(prep_status, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        variant_log = {"status": "started", "splits": {}}
        for split in ["train", "val"]:
            tok_result = tokenize_fn(variant, split, overwrite=overwrite)
            feat_result = extract_features_fn(variant, split, overwrite=overwrite)
            manifest_result = build_manifest_fn(variant, split, overwrite=overwrite)
            variant_log["splits"][split] = {"tokenize": tok_result, "features": feat_result, "manifest": manifest_result}
            prep_status[variant] = variant_log
            prep_status_path.write_text(json.dumps(prep_status, ensure_ascii=False, indent=2), encoding="utf-8")

        variant_log["status"] = "ready"
        variant_log["train_manifest"] = str(train_manifest)
        variant_log["val_manifest"] = str(val_manifest)
        prep_status[variant] = variant_log
        prep_status_path.write_text(json.dumps(prep_status, ensure_ascii=False, indent=2), encoding="utf-8")
    return prep_status


def run_quantity_experiments(
    stage2_root: Path,
    quantity_variants: list[str],
    run_stage2_experiment_fn: Callable[[str], dict],
    allow_overwrite: bool = False,
) -> dict:
    results_path = stage2_root / "quantity_results.json"
    try:
        quantity_results = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        quantity_results = {}

    for variant in quantity_variants:
        if variant in quantity_results and not allow_overwrite:
            continue
        quantity_results[variant] = run_stage2_experiment_fn(variant)
        results_path.write_text(json.dumps(quantity_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return quantity_results


def summarize_quantity_results(stage2_root: Path) -> dict:
    results_path = stage2_root / "quantity_results.json"
    quantity_results = json.loads(results_path.read_text(encoding="utf-8"))
    quantity_ranking = []
    for variant, result in quantity_results.items():
        train_loss = result.get("final_train_mean_loss")
        val_loss = (result.get("val_result") or {}).get("mean_loss")
        qty = int(variant.split("_qty_")[-1])
        quantity_ranking.append({"qty": qty, "variant": variant, "train_loss": train_loss, "val_loss": val_loss})
    quantity_ranking = sorted(quantity_ranking, key=lambda x: x["qty"])

    best_qty_row = min(
        quantity_ranking,
        key=lambda x: x["val_loss"] if x["val_loss"] is not None else float("inf"),
    ) if quantity_ranking else None
    out = {"quantity_ranking": quantity_ranking, "best_quantity_row": best_qty_row}
    out_path = stage2_root / "quantity_results_summary.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
