from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ..common import load_jsonl, write_json


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
    write_json(out_path, out, overwrite=overwrite)
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
    write_json(out_path, out, overwrite=overwrite)
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

    def build_judge_prompt(instruction: str, task_type: str, assistant_a_text: str, assistant_b_text: str) -> str:
        return (
            "You are an impartial evaluator for a vision-language instruction-following project.\n\n"
            "Your job is to compare two candidate responses to the SAME instruction.\n\n"
            "Evaluation criteria:\n"
            "1. Accuracy\n"
            "2. Relevance to the instruction\n"
            "3. Helpfulness\n"
            "4. Completeness / level of detail\n"
            "5. Reasoning quality (if the task requires reasoning)\n\n"
            f"Task type: {task_type}\n\n"
            "Instruction:\n"
            f"{instruction}\n\n"
            "Assistant A:\n"
            f"{assistant_a_text}\n\n"
            "Assistant B:\n"
            f"{assistant_b_text}\n\n"
            "Choose exactly one winner:\n"
            "- assistant_a\n"
            "- assistant_b\n"
            "- tie\n\n"
            "Return JSON only in this format:\n"
            "{\n"
            '  "winner": "assistant_a" or "assistant_b" or "tie",\n'
            '  "reason": "one short sentence"\n'
            "}"
        )

    requests = []
    results_template = []
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
                    "judge_prompt": build_judge_prompt(
                        instruction=instruction,
                        task_type=task_type,
                        assistant_a_text=pair[0][1],
                        assistant_b_text=pair[1][1],
                    ),
                }
            )
            results_template.append({"request_id": request_id, "winner": None, "reason": ""})

    pairwise_md = stage2_root / "pairwise_judge_requests.md"
    pairwise_template = stage2_root / "pairwise_judge_results_template.json"
    pairwise_filled = stage2_root / "pairwise_judge_results_filled.json"
    pairwise_summary = stage2_root / "pairwise_judge_summary.json"

    with pairwise_md.open("w", encoding="utf-8") as f:
        f.write("# Pairwise Judge Requests\n\n")
        f.write(f"Baseline variant: {baseline_variant}\n\n")
        f.write(f"Candidate variants: {', '.join(candidate_variants)}\n\n")
        f.write(f"Total pairwise requests: {len(requests)}\n\n")
        for req in requests:
            f.write(f"## {req['request_id']}\n\n")
            f.write(f"- **image_id:** {req['image_id']}\n")
            f.write(f"- **task_type:** {req['task_type']}\n")
            f.write(f"- **baseline_variant:** {req['baseline_variant']}\n")
            f.write(f"- **candidate_variant:** {req['candidate_variant']}\n")
            f.write(f"- **assistant_a_variant:** {req['assistant_a_variant']}\n")
            f.write(f"- **assistant_b_variant:** {req['assistant_b_variant']}\n\n")
            f.write("**Instruction**\n\n")
            f.write(req["instruction"] + "\n\n")
            f.write("### Assistant A\n\n")
            f.write(req["assistant_a_text"] + "\n\n")
            f.write("### Assistant B\n\n")
            f.write(req["assistant_b_text"] + "\n\n")
            f.write("### Judge Prompt\n\n")
            f.write("```text\n")
            f.write(req["judge_prompt"])
            f.write("\n```\n\n")

    write_json(pairwise_template, results_template, overwrite=True)

    optional_summary = None
    if pairwise_filled.exists():
        filled_results = json.loads(pairwise_filled.read_text(encoding="utf-8"))
        results_by_id = {row["request_id"]: row for row in filled_results}
        candidate_summaries = []
        for candidate in candidate_variants:
            wins = losses = ties = missing = 0
            candidate_requests = [req for req in requests if req["candidate_variant"] == candidate]
            for req in candidate_requests:
                result = results_by_id.get(req["request_id"])
                if result is None:
                    missing += 1
                    continue
                winner = result.get("winner")
                if winner == "tie":
                    ties += 1
                    continue
                if winner == "assistant_a":
                    winner_variant = req["assistant_a_variant"]
                elif winner == "assistant_b":
                    winner_variant = req["assistant_b_variant"]
                else:
                    missing += 1
                    continue
                if winner_variant == candidate:
                    wins += 1
                elif winner_variant == baseline_variant:
                    losses += 1
                else:
                    missing += 1
            decided = wins + losses
            win_rate = wins / decided if decided > 0 else None
            candidate_summaries.append(
                {
                    "candidate_variant": candidate,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "missing_or_invalid": missing,
                    "decided": decided,
                    "win_rate_over_decided": win_rate,
                }
            )
        optional_summary = {"baseline_variant": baseline_variant, "candidate_summaries": candidate_summaries}
        write_json(pairwise_summary, optional_summary, overwrite=True)

    out = {
        "mode": "generated",
        "baseline_variant": baseline_variant,
        "requests": requests,
        "pairwise_requests_md": str(pairwise_md),
        "pairwise_results_template_json": str(pairwise_template),
        "pairwise_results_filled_json": str(pairwise_filled),
        "pairwise_summary_json": str(pairwise_summary),
        "pairwise_summary_generated": optional_summary is not None,
    }
    write_json(out_path, out, overwrite=overwrite)
    return out
