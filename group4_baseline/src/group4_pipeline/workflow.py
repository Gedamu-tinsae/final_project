"""Group4 workflow CLI coordinator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.run_metrics import RunTracker
from src.group4_pipeline.helpers import resolve_group4_config
from src.group4_pipeline.workflow_stages import (
    stage1_preflight,
    stage2_build_plan,
    stage3_build_registry,
    stage3_execute_plan,
    stage4_summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group4 workflow stages from CLI.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--stages", default="all", help="Comma list among 1,2,3,4 or 'all'.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated plan/registry/summary files.")
    p.add_argument("--execute-plan", action="store_true", help="When used with stage 3, execute pending experiments from the plan.")
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


def main() -> int:
    args = parse_args()
    cfg, config_path = resolve_group4_config(PROJECT_ROOT, args.config)
    if not args.allow_non_subset:
        for section in ("required_inputs", "group4_outputs"):
            for key, path in cfg[section].items():
                if args.subset_token not in str(path):
                    raise RuntimeError(
                        f"Refusing non-subset Group4 path for {section}.{key}: {path}. "
                        f"Expected token '{args.subset_token}'. Pass --allow-non-subset to override."
                    )

    enabled = {str(i) for i in range(1, 5)} if args.stages == "all" else {s.strip() for s in args.stages.split(",") if s.strip()}

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CONFIG:", str(config_path) if config_path else "<built-in-defaults>")
    print("overwrite:", args.overwrite)
    tracker = RunTracker(
        group="group4",
        output_root=Path(args.output_root),
        run_name=args.run_name,
        config={"args": vars(args), "project_root": str(PROJECT_ROOT), "config": str(config_path) if config_path else "<built-in-defaults>"},
    )
    print("RUN_DIR:", tracker.run_dir)

    if "1" in enabled:
        with tracker.stage("stage1_preflight") as m:
            print("\n[Stage 1] Preflight dependencies")
            out = stage1_preflight(cfg)
            for k, v in out["status"].items():
                print(f"{v}: {k}")
            if out["missing"]:
                print("blocked_missing:", out["missing"])
            m.update({"missing": len(out["missing"]), "required": len(out["status"])})

    if "2" in enabled:
        with tracker.stage("stage2_plan") as m:
            print("\n[Stage 2] Build experiment plan")
            out = stage2_build_plan(cfg, overwrite=args.overwrite)
            print("plan:", out["write"]["mode"], "->", out["write"]["path"])
            print("num_experiments:", out["num_experiments"])
            m.update({"mode": out["write"]["mode"], "num_experiments": out["num_experiments"]})

    if "3" in enabled:
        with tracker.stage("stage3_registry") as m:
            print("\n[Stage 3] Build run registry")
            out = stage3_build_registry(cfg, overwrite=args.overwrite)
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
                ex = stage3_execute_plan(cfg, args, PROJECT_ROOT)
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
            out = stage4_summarize(cfg, overwrite=args.overwrite)
            if "blocked" in out:
                print("blocked:", out["blocked"])
                m.update({"blocked": out["blocked"]})
            else:
                print("summary_json:", out["summary_json"]["mode"], "->", out["summary_json"]["path"])
                print("summary_md:", out["summary_md"]["mode"], "->", out["summary_md"]["path"])
                if "fig_trainable_params" in out:
                    print("fig_trainable_params:", out["fig_trainable_params"]["mode"], "->", out["fig_trainable_params"]["path"])
                if "fig_runtime" in out:
                    print("fig_runtime:", out["fig_runtime"]["mode"], "->", out["fig_runtime"]["path"])
                if "fig_val_loss" in out:
                    print("fig_val_loss:", out["fig_val_loss"]["mode"], "->", out["fig_val_loss"]["path"])
                print("best_experiment:", out["best_experiment"])
                m.update({"best_experiment": out["best_experiment"], "summary_mode": out["summary_json"]["mode"]})

    summary = tracker.finalize()
    print("Run summary:", summary["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
