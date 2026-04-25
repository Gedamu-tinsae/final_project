#!/usr/bin/env python3
"""Mark completed plan entries in Group4 registry based on existing manual results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile Group4 run registry with existing results.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--default-lora-rank", type=int, default=8)
    p.add_argument("--default-relora-merge-freq", type=int, default=500)
    p.add_argument("--default-sft-budget", type=float, default=1.0)
    p.add_argument("--overwrite-status", action="store_true")
    return p.parse_args()


def _expand_project_root(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    token = "${PROJECT_ROOT}"
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        elif isinstance(v, list):
            out[k] = [x.replace(token, str(project_root)) if isinstance(x, str) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _expand_project_root(v, project_root)
        else:
            out[k] = v
    return out


def _canon_target(t: str | None) -> str:
    target = (t or "").strip().lower()
    if target == "qv_mlp":
        return "all"
    return target or "qv"


def _approx_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


def _matches(
    plan_exp: dict[str, Any],
    result_row: dict[str, Any],
    default_lora_rank: int,
    default_relora_merge_freq: int,
    default_sft_budget: float,
) -> bool:
    method = str(plan_exp.get("method", "")).strip().lower()
    if method != str(result_row.get("method", "")).strip().lower():
        return False

    plan_target = _canon_target(str(plan_exp.get("target_modules", "qv")))
    result_target = _canon_target(str(result_row.get("target_modules", "qv")))
    if plan_target != result_target:
        return False

    if method in {"lora", "relora"}:
        plan_rank = int(plan_exp.get("lora_rank", default_lora_rank))
        result_rank = result_row.get("lora_rank", default_lora_rank)
        try:
            result_rank_i = int(result_rank)
        except Exception:
            result_rank_i = default_lora_rank
        if plan_rank != result_rank_i:
            return False
        if method == "relora":
            plan_mf = int(plan_exp.get("relora_merge_freq", default_relora_merge_freq))
            result_mf = result_row.get("relora_merge_freq", default_relora_merge_freq)
            try:
                result_mf_i = int(result_mf)
            except Exception:
                result_mf_i = default_relora_merge_freq
            return plan_mf == result_mf_i
        return True

    if method == "selective_ft":
        plan_budget = float(plan_exp.get("unfreeze_budget_pct", default_sft_budget))
        result_budget = result_row.get("budget_pct", default_sft_budget)
        try:
            result_budget_f = float(result_budget)
        except Exception:
            result_budget_f = default_sft_budget
        return _approx_eq(plan_budget, result_budget_f)

    return False


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = _expand_project_root(json.loads(cfg_path.read_text(encoding="utf-8")), PROJECT_ROOT)

    plan_path = Path(cfg["group4_outputs"]["plan_json"])
    reg_path = Path(cfg["group4_outputs"]["run_registry_json"])
    res_path = Path(cfg["group4_outputs"]["results_manual_json"])

    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan: {plan_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing registry: {reg_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"Missing manual results: {res_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    manual = json.loads(res_path.read_text(encoding="utf-8"))

    experiments = list(plan.get("experiments", []))
    entries = registry.setdefault("entries", [])
    rows = list(manual.get("results", []))

    entry_by_id: dict[str, dict[str, Any]] = {}
    for e in entries:
        exp_id = str(e.get("experiment_id", "")).strip()
        if exp_id:
            entry_by_id[exp_id] = e

    marked = 0
    skipped_existing = 0
    for exp in experiments:
        exp_id = str(exp.get("experiment_id", "")).strip()
        if not exp_id:
            continue

        matched = None
        for row in rows:
            if _matches(
                exp,
                row,
                args.default_lora_rank,
                args.default_relora_merge_freq,
                args.default_sft_budget,
            ):
                matched = row
                break
        if matched is None:
            continue

        entry = entry_by_id.get(exp_id)
        if entry is None:
            entry = {"experiment_id": exp_id}
            entries.append(entry)
            entry_by_id[exp_id] = entry

        current_status = str(entry.get("status", "pending"))
        if current_status == "completed" and not args.overwrite_status:
            skipped_existing += 1
            continue

        entry["status"] = "completed"
        entry["finished"] = "ok"
        entry["exit_code"] = 0
        entry["method"] = str(exp.get("method", ""))
        entry["source"] = "reconciled_from_results_manual"
        entry["matched_result_experiment_id"] = str(matched.get("experiment_id", ""))
        entry["target_modules"] = str(exp.get("target_modules", ""))
        if entry["method"] in {"lora", "relora"}:
            entry["lora_rank"] = int(exp.get("lora_rank", args.default_lora_rank))
            if entry["method"] == "relora":
                entry["relora_merge_freq"] = int(exp.get("relora_merge_freq", args.default_relora_merge_freq))
        else:
            entry["unfreeze_budget_pct"] = float(exp.get("unfreeze_budget_pct", args.default_sft_budget))
        marked += 1

    registry["num_entries"] = len(entries)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CONFIG:", cfg_path)
    print("PLAN:", plan_path)
    print("REGISTRY:", reg_path)
    print("RESULTS:", res_path)
    print("marked_completed:", marked)
    print("skipped_existing_completed:", skipped_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
