from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_jsonl, write_json


def _safe_mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _safe_median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def build_dataset_quality_diagnostics(stage2_root: Path, all_variants: list[str], overwrite: bool = False) -> dict:
    out_path = stage2_root / "dataset_quality_diagnostics.json"
    if out_path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(out_path)}
    pool_path = stage2_root / "shared_quality_pool.json"
    selected_ids = None
    if pool_path.exists():
        selected_ids = set(json.loads(pool_path.read_text(encoding="utf-8"))["selected_image_ids"])

    diagnostics = []
    for variant in all_variants:
        rows = load_jsonl(stage2_root / variant / "stage2_dataset.jsonl")
        if selected_ids is not None:
            rows = [r for r in rows if r["image_id"] in selected_ids]

        task_buckets = defaultdict(list)
        for row in rows:
            task_buckets[row["task_type"]].append(row)

        response_counter = Counter(str(r.get("response", "")).strip() for r in rows)
        duplicate_response_count = sum(1 for resp, c in response_counter.items() if resp and c > 1)
        overall_prompt_words = [len(str(r.get("instruction", "")).split()) for r in rows]
        overall_response_words = [len(str(r.get("response", "")).split()) for r in rows]

        variant_summary = {
            "variant": variant,
            "num_rows": len(rows),
            "num_unique_images": len({r["image_id"] for r in rows}),
            "duplicate_response_count": duplicate_response_count,
            "overall": {
                "mean_instruction_words": _safe_mean(overall_prompt_words),
                "median_instruction_words": _safe_median(overall_prompt_words),
                "mean_response_words": _safe_mean(overall_response_words),
                "median_response_words": _safe_median(overall_response_words),
            },
            "by_task": {},
        }
        for task_type, task_rows in sorted(task_buckets.items()):
            iw = [len(str(r.get("instruction", "")).split()) for r in task_rows]
            rw = [len(str(r.get("response", "")).split()) for r in task_rows]
            variant_summary["by_task"][task_type] = {
                "num_rows": len(task_rows),
                "num_unique_images": len({r["image_id"] for r in task_rows}),
                "mean_instruction_words": _safe_mean(iw),
                "mean_response_words": _safe_mean(rw),
            }
        diagnostics.append(variant_summary)

    out = {"mode": "generated", "all_variants": all_variants, "diagnostics": diagnostics}
    write_json(out_path, out, overwrite=True)
    return out


def build_qualitative_samples_pack(
    stage2_root: Path,
    all_variants: list[str],
    per_task: int = 4,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    out_path = stage2_root / "qualitative_comparison_samples.json"
    if out_path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(out_path)}
    rows_by_variant_key: dict[str, dict] = {}
    for variant in all_variants:
        rows = load_jsonl(stage2_root / variant / "stage2_dataset.jsonl")
        rows_by_variant_key[variant] = {(r["image_id"], r["task_type"]): r for r in rows}

    common_keys = None
    for variant in all_variants:
        keys = set(rows_by_variant_key[variant].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = sorted(common_keys or [])

    aligned_keys = []
    for key in common_keys:
        instructions = [rows_by_variant_key[v][key]["instruction"] for v in all_variants]
        if len(set(instructions)) == 1:
            aligned_keys.append(key)

    keys_by_task = defaultdict(list)
    for key in aligned_keys:
        keys_by_task[key[1]].append(key)

    rng = random.Random(seed)
    selected_keys = []
    for _, keys in sorted(keys_by_task.items()):
        pool = list(keys)
        rng.shuffle(pool)
        selected_keys.extend(pool[:per_task])

    samples = []
    for image_id, task_type in selected_keys:
        base_row = rows_by_variant_key[all_variants[0]][(image_id, task_type)]
        sample = {
            "image_id": image_id,
            "task_type": task_type,
            "instruction": base_row["instruction"],
            "responses": {v: rows_by_variant_key[v][(image_id, task_type)]["response"] for v in all_variants},
        }
        samples.append(sample)

    out = {"mode": "generated", "all_variants": all_variants, "num_samples": len(samples), "samples": samples}
    write_json(out_path, out, overwrite=True)
    return out


def build_pairwise_judge_requests(
    stage2_root: Path,
    baseline_variant: str,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict:
    out_path = stage2_root / "pairwise_judge_requests.json"
    if out_path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(out_path)}
    eval_pack_path = stage2_root / "heldout_eval_pack.json"
    eval_pack = json.loads(eval_pack_path.read_text(encoding="utf-8"))
    all_variants = eval_pack["all_variants"]
    candidate_variants = [v for v in all_variants if v != baseline_variant]
    rng = random.Random(seed)

    requests = []
    for i, sample in enumerate(eval_pack["samples"], start=1):
        instruction = sample["instruction"]
        task_type = sample["task_type"]
        baseline_response = sample["reference_responses"][baseline_variant]
        for candidate in candidate_variants:
            candidate_response = sample["reference_responses"][candidate]
            pair = [(baseline_variant, baseline_response), (candidate, candidate_response)]
            rng.shuffle(pair)
            request_id = f"{candidate}__sample_{i:03d}"
            requests.append(
                {
                    "request_id": request_id,
                    "image_id": sample["image_id"],
                    "task_type": task_type,
                    "instruction": instruction,
                    "baseline_variant": baseline_variant,
                    "candidate_variant": candidate,
                    "assistant_a_variant": pair[0][0],
                    "assistant_a_text": pair[0][1],
                    "assistant_b_variant": pair[1][0],
                    "assistant_b_text": pair[1][1],
                }
            )

    out = {"mode": "generated", "baseline_variant": baseline_variant, "requests": requests}
    write_json(out_path, out, overwrite=True)
    return out
