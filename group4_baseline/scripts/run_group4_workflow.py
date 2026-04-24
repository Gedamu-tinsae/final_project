"""CLI orchestration for Group4 (parameter-efficient tuning track).

Safe by default:
- verifies prerequisites from Group1/Group2
- generates a reproducible experiment plan and run registry
- summarizes collected results

Optional execution mode:
- can execute planned experiments by calling run_group4_peft_smoke.py
- updates run registry statuses
- auto-ingests per-run metrics into results_manual_json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.run_metrics import RunTracker


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group4 workflow stages from CLI.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--stages", default="all", help="Comma list among 1,2,3,4 or 'all'.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated plan/registry/summary files.")
    p.add_argument(
        "--execute-plan",
        action="store_true",
        help="When used with stage 3, execute pending experiments from the plan.",
    )
    p.add_argument("--max-experiments", type=int, default=0, help="Max experiments to run when --execute-plan is set (0=all).")
    p.add_argument("--max-rows", type=int, default=64, help="Rows passed to run_group4_peft_smoke.py.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-overwrite-experiment-outputs", action="store_true")
    p.add_argument("--val-every-steps", type=int, default=200)
    p.add_argument("--val-max-batches", type=int, default=0)
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--run-name", default="group4_workflow")
    p.add_argument("--subset-token", default="subset_10000_seed42")
    p.add_argument("--allow-non-subset", action="store_true")
    return p.parse_args()


def _write_json_guarded(path: Path, payload: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"mode": "generated", "path": str(path)}


def _stage1_preflight(cfg: dict[str, Any]) -> dict[str, Any]:
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


def _stage2_build_plan(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    exp = cfg["experiment_space"]
    methods = list(exp["methods"])
    lora_ranks = list(exp["lora_ranks"])
    targets = list(exp["target_modules"])
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
    write = _write_json_guarded(out, plan, overwrite=overwrite)
    return {"write": write, "num_experiments": len(experiments)}


def _stage3_build_registry(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
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
    write = _write_json_guarded(out, registry, overwrite=overwrite)
    return {"write": write, "num_entries": len(entries)}


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"num_entries": 0, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_experiment_cli_args(exp: dict[str, Any], args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[str, list[str], str]:
    method = str(exp.get("method", ""))
    exp_id = str(exp.get("experiment_id", ""))
    if method not in {"lora", "selective_ft"}:
        raise ValueError(f"Unsupported method in experiment {exp_id}: {method}")

    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_group4_peft_smoke.py"),
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
        "--subset-token",
        args.subset_token,
    ]
    if args.allow_non_subset:
        cmd.append("--allow-non-subset")
    if args.allow_overwrite_experiment_outputs:
        cmd.append("--overwrite")

    if method == "lora":
        target = str(exp.get("target_modules", "qv"))
        lora_variant = "qv"
        if target == "all":
            lora_variant = "all_weights"
        cmd.extend(["--lora-variant", lora_variant, "--target-modules", target])
    else:
        target = str(exp.get("target_modules", "qv"))
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


def _stage3_execute_plan(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(cfg["group4_outputs"]["plan_json"])
    registry_path = Path(cfg["group4_outputs"]["run_registry_json"])
    results_path = Path(cfg["group4_outputs"]["results_manual_json"])
    if not plan_path.exists():
        return {"blocked": f"missing {plan_path}"}

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = _load_registry(registry_path)
    entries = registry.setdefault("entries", [])
    if not entries:
        # build baseline registry in-memory if missing
        for exp in plan.get("experiments", []):
            entries.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "status": "pending",
                    "output_dir": f"{cfg['project_root']}/artifacts/group4/{exp.get('experiment_id','')}",
                    "notes": "Populated by stage3 execution.",
                }
            )

    by_id = {str(e.get("experiment_id", "")): e for e in entries}
    executed = 0
    succeeded = 0
    failed = 0
    for exp in plan.get("experiments", []):
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
        exp_id_, cmd, method = _build_experiment_cli_args(exp, args, cfg)
        print(f"execute: {exp_id_} method={method}")
        print("cmd:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        executed += 1
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
    results_exists = results_path.exists()
    return {
        "mode": "executed",
        "registry_path": str(registry_path),
        "results_manual_json_exists": bool(results_exists),
        "executed": executed,
        "succeeded": succeeded,
        "failed": failed,
    }


def _stage4_summarize(cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    in_path = Path(cfg["group4_outputs"]["results_manual_json"])
    if not in_path.exists():
        return {"blocked": f"missing {in_path}"}

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = raw.get("results", [])
    if not rows:
        return {"blocked": f"{in_path} has no rows in results[]"}

    # Lower val_loss + trainable params is better; higher win_rate is better.
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
    write_json = _write_json_guarded(out_json, summary, overwrite=overwrite)

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

    return {"summary_json": write_json, "summary_md": write_md, "best_experiment": best.get("experiment_id", "")}


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(raw_cfg, PROJECT_ROOT)
    if not args.allow_non_subset:
        for section in ("required_inputs", "group4_outputs"):
            for key, path in cfg[section].items():
                if args.subset_token not in str(path):
                    raise RuntimeError(
                        f"Refusing non-subset Group4 path for {section}.{key}: {path}. "
                        f"Expected token '{args.subset_token}'. Pass --allow-non-subset to override."
                    )

    if args.stages == "all":
        enabled = {str(i) for i in range(1, 5)}
    else:
        enabled = {s.strip() for s in args.stages.split(",") if s.strip()}

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CONFIG:", config_path)
    print("overwrite:", args.overwrite)
    tracker = RunTracker(
        group="group4",
        output_root=Path(args.output_root),
        run_name=args.run_name,
        config={"args": vars(args), "project_root": str(PROJECT_ROOT), "config": str(config_path)},
    )
    print("RUN_DIR:", tracker.run_dir)

    if "1" in enabled:
        with tracker.stage("stage1_preflight") as m:
            print("\n[Stage 1] Preflight dependencies")
            out = _stage1_preflight(cfg)
            for k, v in out["status"].items():
                print(f"{v}: {k}")
            if out["missing"]:
                print("blocked_missing:", out["missing"])
            m.update({"missing": len(out["missing"]), "required": len(out["status"])})

    if "2" in enabled:
        with tracker.stage("stage2_plan") as m:
            print("\n[Stage 2] Build experiment plan")
            out = _stage2_build_plan(cfg, overwrite=args.overwrite)
            print("plan:", out["write"]["mode"], "->", out["write"]["path"])
            print("num_experiments:", out["num_experiments"])
            m.update({"mode": out["write"]["mode"], "num_experiments": out["num_experiments"]})

    if "3" in enabled:
        with tracker.stage("stage3_registry") as m:
            print("\n[Stage 3] Build run registry")
            out = _stage3_build_registry(cfg, overwrite=args.overwrite)
            if "blocked" in out:
                print("blocked:", out["blocked"])
                m.update({"blocked": out["blocked"]})
            else:
                print("registry:", out["write"]["mode"], "->", out["write"]["path"])
                print("num_entries:", out["num_entries"])
                m.update({"mode": out["write"]["mode"], "num_entries": out["num_entries"]})
        if args.execute_plan:
            with tracker.stage("stage3_execute_plan") as m:
                print("\n[Stage 3b] Execute plan")
                ex = _stage3_execute_plan(cfg, args)
                if "blocked" in ex:
                    print("blocked:", ex["blocked"])
                    m.update({"blocked": ex["blocked"]})
                else:
                    print(
                        "executed:",
                        f"count={ex['executed']}",
                        f"succeeded={ex['succeeded']}",
                        f"failed={ex['failed']}",
                        f"results_manual_json_exists={ex['results_manual_json_exists']}",
                    )
                    m.update(
                        {
                            "executed": ex["executed"],
                            "succeeded": ex["succeeded"],
                            "failed": ex["failed"],
                            "results_manual_json_exists": ex["results_manual_json_exists"],
                        }
                    )

    if "4" in enabled:
        with tracker.stage("stage4_summarize") as m:
            print("\n[Stage 4] Summarize results")
            out = _stage4_summarize(cfg, overwrite=args.overwrite)
            if "blocked" in out:
                print("blocked:", out["blocked"])
                m.update({"blocked": out["blocked"]})
            else:
                print("summary_json:", out["summary_json"]["mode"], "->", out["summary_json"]["path"])
                print("summary_md:", out["summary_md"]["mode"], "->", out["summary_md"]["path"])
                print("best_experiment:", out["best_experiment"])
                m.update({"best_experiment": out["best_experiment"], "summary_mode": out["summary_json"]["mode"]})

    summary = tracker.finalize()
    print("Run summary:", summary["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
