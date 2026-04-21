from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import load_jsonl


def audit_stage2_variants(stage2_root: Path, all_variants: list[str]) -> dict:
    required_base = {"image", "image_id", "generator_model", "task_type", "instruction", "response"}
    allowed_tasks = {"conversation", "detailed_description", "complex_reasoning"}

    variant_image_sets: dict[str, set] = {}
    variant_task_counts: dict[str, dict] = {}
    variant_row_counts: dict[str, int] = {}
    issues: dict[str, list[str]] = {}

    for variant in all_variants:
        dataset_path = stage2_root / variant / "stage2_dataset.jsonl"
        if not dataset_path.exists():
            issues.setdefault(variant, []).append(f"missing dataset: {dataset_path}")
            continue
        rows = load_jsonl(dataset_path)
        bad_rows = 0
        task_counts = Counter(r.get("task_type", "UNKNOWN") for r in rows)
        images = {r["image_id"] for r in rows if "image_id" in r}

        for row in rows:
            missing = required_base - set(row.keys())
            if missing:
                bad_rows += 1
                continue
            if row["task_type"] not in allowed_tasks:
                bad_rows += 1
            if row["task_type"] == "conversation" and ("history" not in row or "turn_index" not in row):
                bad_rows += 1

        variant_image_sets[variant] = images
        variant_task_counts[variant] = dict(task_counts)
        variant_row_counts[variant] = len(rows)
        if bad_rows:
            issues.setdefault(variant, []).append(f"bad rows: {bad_rows}")

    consistency: dict[str, bool] = {}
    if all_variants:
        baseline = all_variants[0]
        base_images = variant_image_sets.get(baseline, set())
        for variant in all_variants[1:]:
            consistency[variant] = variant_image_sets.get(variant, set()) == base_images

    return {
        "variant_row_counts": variant_row_counts,
        "variant_task_counts": variant_task_counts,
        "issues": issues,
        "image_set_consistency_against_baseline": consistency,
    }

