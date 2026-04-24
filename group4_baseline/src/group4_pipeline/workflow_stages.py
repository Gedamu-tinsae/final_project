"""Stage implementations for Group4 workflow orchestration."""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_json_guarded(path: Path, payload: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"mode": "generated", "path": str(path)}


def _canonical_target(target: str) -> str:
    t = str(target).strip().lower()
    if t == "qv_mlp":
        return "all"
    if t not in {"qv", "all"}:
        raise ValueError(f"Unsupported target_modules value: {target}")
    return t


def _csv_set(raw: str) -> set[str]:
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def _try_plot_metric_bars(
    out_png: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {"mode": "skipped_no_matplotlib", "path": str(out_png)}

    labels: list[str] = []
    vals: list[float] = []
    for r in rows:
        raw = r.get(metric_key, None)
        if raw is None:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        labels.append(str(r.get("experiment_id", r.get("method", "run"))))
        vals.append(v)

    if not labels:
        return {"mode": "skipped_no_values", "path": str(out_png), "metric": metric_key}

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 1.1), 4))
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return {"mode": "generated", "path": str(out_png), "metric": metric_key, "count": len(labels)}


def stage1_preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    status: dict[str, str] = {}
    for key, value in cfg["required_inputs"].items():
        p = Path(value)
        if p.exists():
            status[key] = "OK"
        else:
            status[key] = "MISSING"
            missing.append(key)
    return {"status": status, "missing": missing}


def stage2_build_plan(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    exp = cfg["experiment_space"]
    methods = list(exp["methods"])
    lora_ranks = list(exp["lora_ranks"])
    # Deduplicate while preserving order.
    raw_targets = [str(t) for t in exp["target_modules"]]
    targets: list[str] = []
    for tt in raw_targets:
        ct = _canonical_target(tt)
        if ct not in targets:
            targets.append(ct)
    selective_budgets = list(exp["selective_ft_budget_pct"])
    seed = int(exp["seed"])
    train_steps = int(exp["train_budget_steps"])

    experiments: list[dict[str, Any]] = []
    exp_id = 1
    for method in methods:
        if method == "lora":
            for rank, target in itertools.product(lora_ranks, targets):
                experiments.append(
                    {
                        "experiment_id": f"g4_lora_{exp_id:03d}",
                        "method": "lora",
                        "lora_rank": int(rank),
                        "target_modules": str(target),
                        "train_budget_steps": train_steps,
                        "seed": seed,
                    }
                )
                exp_id += 1
        elif method == "selective_ft":
            for pct, target in itertools.product(selective_budgets, targets):
                experiments.append(
                    {
                        "experiment_id": f"g4_sft_{exp_id:03d}",
                        "method": "selective_ft",
                        "unfreeze_budget_pct": float(pct),
                        "target_modules": str(target),
                        "train_budget_steps": train_steps,
                        "seed": seed,
                    }
                )
                exp_id += 1
        else:
            raise ValueError(f"Unsupported method: {method}")

    plan = {
        "project_root": cfg["project_root"],
        "group1_root": cfg["group1_root"],
        "group2_root": cfg["group2_root"],
        "num_experiments": len(experiments),
        "experiments": experiments,
    }
    out = Path(cfg["group4_outputs"]["plan_json"])
    write = write_json_guarded(out, plan, overwrite=overwrite)
    return {"write": write, "num_experiments": len(experiments)}


def stage3_build_registry(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    plan_path = Path(cfg["group4_outputs"]["plan_json"])
    if not plan_path.exists():
        return {"blocked": f"missing {plan_path}"}

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for exp in plan["experiments"]:
        exp_id = exp["experiment_id"]
        output_dir = f"{cfg['project_root']}/artifacts/group4/{exp_id}"
        entries.append(
            {
                "experiment_id": exp_id,
                "status": "pending",
                "output_dir": output_dir,
                "notes": "Populated by stage3 execution when --execute-plan is used.",
            }
        )

    registry = {
        "num_entries": len(entries),
        "entries": entries,
    }
    out = Path(cfg["group4_outputs"]["run_registry_json"])
    write = write_json_guarded(out, registry, overwrite=overwrite)
    return {"write": write, "num_entries": len(entries)}


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"num_entries": 0, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_experiment_cli_args(exp: dict[str, Any], args, project_root: Path) -> tuple[str, list[str], str]:
    method = str(exp.get("method", ""))
    exp_id = str(exp.get("experiment_id", ""))
    if method not in {"lora", "selective_ft"}:
        raise ValueError(f"Unsupported method in experiment {exp_id}: {method}")

    cmd: list[str] = [
        sys.executable,
        str(project_root / "scripts" / "run_group4_peft_smoke.py"),
        "--config",
        str(Path(args.config)),
        "--method",
        method,
        "--max-rows",
        str(int(args.max_rows)),
        "--batch-size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--dtype",
        args.dtype,
        "--seed",
        str(int(args.seed)),
        "--append-manual-results",
        "--val-every-steps",
        str(int(args.val_every_steps)),
        "--val-max-batches",
        str(int(args.val_max_batches)),
        "--run-name",
        f"group4_peft_{exp_id}",
        "--experiment-id",
        exp_id,
        "--subset-token",
        args.subset_token,
    ]
    if args.allow_non_subset:
        cmd.append("--allow-non-subset")
    if args.allow_overwrite_experiment_outputs:
        cmd.append("--overwrite")

    if method == "lora":
        target = _canonical_target(str(exp.get("target_modules", "qv")))
        lora_variant = "qv"
        if target == "all":
            lora_variant = "all_weights"
        cmd.extend(["--lora-variant", lora_variant, "--target-modules", target])
    else:
        target = _canonical_target(str(exp.get("target_modules", "qv")))
        budget = float(exp.get("unfreeze_budget_pct", 1.0))
        cmd.extend(
            [
                "--target-modules",
                target,
                "--selection-strategy",
                "magnitude",
                "--budget-pct",
                str(budget),
            ]
        )

    return exp_id, cmd, method


def stage3_execute_plan(cfg: dict[str, Any], args, project_root: Path) -> dict[str, Any]:
    plan_path = Path(cfg["group4_outputs"]["plan_json"])
    registry_path = Path(cfg["group4_outputs"]["run_registry_json"])
    results_path = Path(cfg["group4_outputs"]["results_manual_json"])
    if not plan_path.exists():
        return {"blocked": f"missing {plan_path}"}

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = _load_registry(registry_path)
    entries = registry.setdefault("entries", [])
    if not entries:
        for exp in plan.get("experiments", []):
            entries.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "status": "pending",
                    "output_dir": f"{cfg['project_root']}/artifacts/group4/{exp.get('experiment_id','')}",
                    "notes": "Populated by stage3 execution.",
                }
            )

    requested_ids = _csv_set(getattr(args, "plan_experiment_ids", ""))
    requested_methods = {m.lower() for m in _csv_set(getattr(args, "plan_methods", ""))}
    requested_targets = {_canonical_target(t) for t in _csv_set(getattr(args, "plan_target_modules", ""))}
    requested_lora_ranks = {int(x) for x in _csv_set(getattr(args, "plan_lora_ranks", ""))}
    requested_sft_budgets = {float(x) for x in _csv_set(getattr(args, "plan_sft_budgets", ""))}

    selected_experiments: list[dict[str, Any]] = []
    for exp in plan.get("experiments", []):
        exp_id = str(exp.get("experiment_id", ""))
        method = str(exp.get("method", "")).strip().lower()
        target = _canonical_target(str(exp.get("target_modules", "qv")))
        if requested_ids and exp_id not in requested_ids:
            continue
        if requested_methods and method not in requested_methods:
            continue
        if requested_targets and target not in requested_targets:
            continue
        if method == "lora" and requested_lora_ranks:
            if int(exp.get("lora_rank", -1)) not in requested_lora_ranks:
                continue
        if method == "selective_ft" and requested_sft_budgets:
            if float(exp.get("unfreeze_budget_pct", -1.0)) not in requested_sft_budgets:
                continue
        selected_experiments.append(exp)

    by_id = {str(e.get("experiment_id", "")): e for e in entries}
    executed = 0
    succeeded = 0
    failed = 0
    for exp in selected_experiments:
        exp_id = str(exp.get("experiment_id", ""))
        if not exp_id:
            continue
        reg = by_id.setdefault(exp_id, {"experiment_id": exp_id, "status": "pending"})
        if str(reg.get("status", "pending")) == "completed" and not args.allow_overwrite_experiment_outputs:
            continue
        if args.max_experiments > 0 and executed >= int(args.max_experiments):
            break

        reg["status"] = "running"
        _save_registry(registry_path, registry)
        exp_id_, cmd, method = _build_experiment_cli_args(exp, args, project_root)
        print(f"execute: {exp_id_} method={method}")
        print("cmd:", " ".join(cmd))
        retries = max(0, int(getattr(args, "plan_retries", 2)))
        retry_sleep_sec = max(1, int(getattr(args, "plan_retry_sleep_sec", 20)))
        proc = None
        attempt = 0
        child_env = os.environ.copy()
        child_env.pop("RUN_METRICS_DISABLE_TPU_SAMPLER", None)
        while True:
            attempt += 1
            proc = subprocess.run(cmd, cwd=str(project_root), env=child_env)
            if proc.returncode == 0 or attempt > (retries + 1):
                break
            print(
                f"retry: experiment={exp_id_} attempt={attempt}/{retries + 1} "
                f"sleep_sec={retry_sleep_sec} returncode={proc.returncode}"
            )
            time.sleep(retry_sleep_sec)
        executed += 1
        reg["attempts"] = int(attempt)
        reg["exit_code"] = int(proc.returncode)
        reg["method"] = method
        reg["finished"] = "ok" if proc.returncode == 0 else "error"
        if proc.returncode == 0:
            reg["status"] = "completed"
            succeeded += 1
        else:
            reg["status"] = "failed"
            failed += 1
        _save_registry(registry_path, registry)

    registry["num_entries"] = len(registry.get("entries", []))
    _save_registry(registry_path, registry)
    return {
        "mode": "executed",
        "registry_path": str(registry_path),
        "results_manual_json_exists": bool(results_path.exists()),
        "selected": len(selected_experiments),
        "total_plan_experiments": len(plan.get("experiments", [])),
        "executed": executed,
        "succeeded": succeeded,
        "failed": failed,
    }


def stage4_summarize(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    in_path = Path(cfg["group4_outputs"]["results_manual_json"])
    if not in_path.exists():
        return {"blocked": f"missing {in_path}"}

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = raw.get("results", [])
    if not rows:
        return {"blocked": f"{in_path} has no rows in results[]"}

    ranked: list[dict[str, Any]] = []
    for row in rows:
        val_loss = float(row.get("val_loss", math.inf))
        win_rate = float(row.get("win_rate_vs_baseline", 0.0))
        trainable_m = float(row.get("trainable_params_millions", math.inf))
        score = (100.0 * win_rate) - (10.0 * val_loss) - (0.2 * trainable_m)
        ranked.append({**row, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    summary = {
        "num_results": len(ranked),
        "best_experiment": best,
        "ranking": ranked,
    }

    out_json = Path(cfg["group4_outputs"]["summary_json"])
    write_json = write_json_guarded(out_json, summary, overwrite=overwrite)

    out_md = Path(cfg["group4_outputs"]["summary_md"])
    if out_md.exists() and not overwrite:
        write_md = {"mode": "skipped_existing", "path": str(out_md)}
    else:
        lines = [
            "# Group4 Results Summary",
            "",
            f"- total_results: {len(ranked)}",
            f"- best_experiment: {best.get('experiment_id', '<unknown>')}",
            f"- best_score: {best['score']:.4f}",
            "",
            "## Ranking",
            "",
            "| rank | experiment_id | method | win_rate_vs_baseline | val_loss | trainable_params_millions | score |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for i, row in enumerate(ranked, start=1):
            lines.append(
                f"| {i} | {row.get('experiment_id','')} | {row.get('method','')} | "
                f"{float(row.get('win_rate_vs_baseline', 0.0)):.4f} | {float(row.get('val_loss', math.inf)):.4f} | "
                f"{float(row.get('trainable_params_millions', math.inf)):.4f} | {row['score']:.4f} |"
            )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_md = {"mode": "generated", "path": str(out_md)}

    # Method comparison charts for presentation.
    base_dir = out_json.parent
    fig_params = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_trainable_params.png",
        ranked,
        "trainable_params_millions",
        "Group4 Method Comparison: Trainable Params (M)",
        "params (millions)",
    )
    fig_runtime = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_runtime_sec.png",
        ranked,
        "wall_time_sec",
        "Group4 Method Comparison: Runtime",
        "seconds",
    )
    fig_val = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_val_loss.png",
        ranked,
        "val_loss",
        "Group4 Method Comparison: Validation Loss",
        "val_loss",
    )
    fig_train_last = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_train_loss_last.png",
        ranked,
        "smoke_loss_last",
        "Group4 Method Comparison: Final Train Loss",
        "train loss (last)",
    )
    fig_steps_per_sec = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_steps_per_sec.png",
        ranked,
        "steps_per_sec",
        "Group4 Method Comparison: Throughput (steps/sec)",
        "steps/sec",
    )
    fig_samples_per_sec = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_samples_per_sec.png",
        ranked,
        "samples_per_sec",
        "Group4 Method Comparison: Throughput (samples/sec)",
        "samples/sec",
    )
    fig_gpu_mem = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_gpu_mem_max_mb.png",
        ranked,
        "gpu_mem_used_max_mb",
        "Group4 Method Comparison: Peak GPU Memory",
        "MB",
    )
    fig_tpu_mem = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_tpu_mem_max_mb.png",
        ranked,
        "tpu_mem_used_max_mb",
        "Group4 Method Comparison: Peak TPU Memory",
        "MB",
    )
    fig_rss_kb = _try_plot_metric_bars(
        base_dir / "fig_method_comparison_rss_kb_max.png",
        ranked,
        "rss_kb_max",
        "Group4 Method Comparison: Peak RSS",
        "KB",
    )

    return {
        "summary_json": write_json,
        "summary_md": write_md,
        "best_experiment": best.get("experiment_id", ""),
        "fig_trainable_params": fig_params,
        "fig_runtime": fig_runtime,
        "fig_val_loss": fig_val,
        "fig_train_loss_last": fig_train_last,
        "fig_steps_per_sec": fig_steps_per_sec,
        "fig_samples_per_sec": fig_samples_per_sec,
        "fig_gpu_mem_max_mb": fig_gpu_mem,
        "fig_tpu_mem_max_mb": fig_tpu_mem,
        "fig_rss_kb_max": fig_rss_kb,
    }
