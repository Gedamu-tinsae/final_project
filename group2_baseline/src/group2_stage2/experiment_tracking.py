from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def select_next_variant(results_path: Path, expected_variants: list[str], current_variant: str | None, allow_overwrite: bool) -> dict:
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        all_results = {}
        results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    missing_variants = [v for v in expected_variants if v not in all_results]
    selected = current_variant or (missing_variants[0] if missing_variants else None)

    if selected is not None and selected not in expected_variants:
        raise ValueError(f"current variant '{selected}' is not in expected variants: {expected_variants}")
    if selected is not None and selected in all_results and not allow_overwrite:
        raise ValueError(f"Result for '{selected}' already exists in {results_path}.")

    return {"all_results": all_results, "missing_variants": missing_variants, "selected_variant": selected}


def run_and_store_variant(
    results_path: Path,
    all_results: dict,
    selected_variant: str | None,
    run_stage2_experiment_fn: Callable[[str], dict],
) -> dict:
    if selected_variant is None:
        return {"all_results": all_results, "ran_variant": None}
    result = run_stage2_experiment_fn(selected_variant)
    all_results[selected_variant] = result
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"all_results": all_results, "ran_variant": selected_variant}


def prompt_alignment_audit(stage2_root: Path, all_variants: list[str], prompt_reference_variant: str) -> dict:
    pool_path = stage2_root / "shared_quality_pool.json"
    selected_ids = None
    if pool_path.exists():
        selected_ids = set(json.loads(pool_path.read_text(encoding="utf-8"))["selected_image_ids"])

    def load_variant_prompt_map(variant: str):
        prompt_map = {}
        duplicates = []
        with (stage2_root / variant / "stage2_dataset.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if selected_ids is not None and row["image_id"] not in selected_ids:
                    continue
                key = (row["image_id"], row["task_type"])
                if key in prompt_map:
                    duplicates.append(key)
                prompt_map[key] = row["instruction"]
        return prompt_map, duplicates

    reference_map, _ = load_variant_prompt_map(prompt_reference_variant)
    ref_keys = set(reference_map.keys())
    rows = []
    mismatch_examples = []

    for variant in all_variants:
        prompt_map, duplicates = load_variant_prompt_map(variant)
        variant_keys = set(prompt_map.keys())
        shared_keys = sorted(ref_keys & variant_keys)
        mismatch_count = 0
        for key in shared_keys:
            if reference_map[key] != prompt_map[key]:
                mismatch_count += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "variant": variant,
                            "image_id": key[0],
                            "task_type": key[1],
                            "reference_instruction": reference_map[key],
                            "variant_instruction": prompt_map[key],
                        }
                    )
        rows.append(
            {
                "variant": variant,
                "num_keys": len(prompt_map),
                "duplicate_key_count": len(duplicates),
                "missing_vs_reference_count": len(ref_keys - variant_keys),
                "extra_vs_reference_count": len(variant_keys - ref_keys),
                "instruction_mismatch_count": mismatch_count,
            }
        )

    out = {"reference_variant": prompt_reference_variant, "audit_rows": rows, "mismatch_examples": mismatch_examples}
    (stage2_root / "prompt_alignment_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _first_existing(d: dict, keys: list[str]):
    for k in keys:
        if d.get(k) is not None:
            return d[k], k
    return None, None


def build_engine_comparison_summary(stage2_root: Path, results_path: Path, expected_variants: list[str]) -> dict:
    all_results = json.loads(results_path.read_text(encoding="utf-8"))
    missing_variants = [v for v in expected_variants if v not in all_results]
    if missing_variants:
        raise ValueError(f"Missing variants in results file: {missing_variants}")

    rows = []
    for variant in expected_variants:
        result = all_results[variant]
        train_value = result.get("final_train_mean_loss")
        val_result = result.get("val_result", {})
        val_value, val_key = _first_existing(
            val_result,
            ["final_val_mean_loss", "final_overall_val_mean_loss", "overall_val_mean_loss", "val_mean_loss", "mean_loss"],
        )
        if val_value is None:
            raise ValueError(f"Could not find validation metric for variant '{variant}'")
        rows.append(
            {
                "variant": variant,
                "final_train_mean_loss": train_value,
                "final_val_mean_loss": val_value,
                "val_key_used": f"val_result.{val_key}",
            }
        )

    rows = sorted(rows, key=lambda x: x["final_val_mean_loss"])
    summary = {"ranking_metric": "final_val_mean_loss_lower_is_better", "best_variant": rows[0]["variant"], "variant_summaries": rows}
    (stage2_root / "engine_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_baseline_relative_comparison(stage2_root: Path, results_path: Path, baseline_variant: str, expected_variants: list[str]) -> dict:
    all_results = json.loads(results_path.read_text(encoding="utf-8"))
    baseline_train = all_results[baseline_variant].get("final_train_mean_loss")
    baseline_val = (all_results[baseline_variant].get("val_result") or {}).get("mean_loss")
    rows = []

    for variant in expected_variants:
        result = all_results.get(variant, {})
        train_loss = result.get("final_train_mean_loss")
        val_loss = (result.get("val_result") or {}).get("mean_loss")
        train_delta = None if baseline_train is None or train_loss is None else (train_loss - baseline_train)
        val_delta = None if baseline_val is None or val_loss is None else (val_loss - baseline_val)
        rows.append(
            {
                "variant": variant,
                "final_train_mean_loss": train_loss,
                "final_val_mean_loss": val_loss,
                "train_delta_vs_baseline": train_delta,
                "val_delta_vs_baseline": val_delta,
            }
        )

    out = {"baseline_variant": baseline_variant, "expected_variants": expected_variants, "rows": rows}
    (stage2_root / "baseline_relative_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

