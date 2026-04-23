"""CLI orchestration for Group2 baseline workflow (notebook-equivalent stages).

This script mirrors notebook stages with guarded/default-safe behavior:
- overwrite is off by default
- stage2 prep defaults to baseline variant + val split first
- model-dependent experiment execution is not forced automatically
"""

from __future__ import annotations

import argparse
import json
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
from src.group2_stage2.bootstrap_runtime import create_stage2_runtime_objects
from src.group2_stage2.data.audit import audit_stage2_variants
from src.group2_stage2.data.manifests import build_stage2_manifest
from src.group2_stage2.data.pipeline import prepare_stage2_variant_splits
from src.group2_stage2.data.splits import build_shared_quality_pool, materialize_train_val_split
from src.group2_stage2.data.tokenization import tokenize_stage2_variant
from src.group2_stage2.data.features import extract_stage2_features
from src.group2_stage2.eval.evaluation_pack import build_heldout_eval_pack
from src.group2_stage2.eval.quality_eval import (
    build_dataset_quality_diagnostics,
    build_pairwise_judge_requests,
    build_qualitative_samples_pack,
)
from src.group2_stage2.eval.reporting import build_engine_plots_and_table
from src.group2_stage2.experiments.experiment_tracking import (
    build_baseline_relative_comparison,
    build_engine_comparison_summary,
    run_and_store_variant,
    prompt_alignment_audit,
    select_next_variant,
)
from src.group2_stage2.experiments.quantity_ablation import (
    build_quantity_variants,
    derive_quantity_plan,
    prepare_quantity_variants,
    register_quantity_variants,
    run_quantity_experiments,
    summarize_quantity_results,
)
from src.group2_stage2.experiments.stage2_experiment_runner import run_stage2_experiment


def _expand_project_root(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    token = "${PROJECT_ROOT}"
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        elif isinstance(v, list):
            out[k] = [x.replace(token, str(project_root)) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group2 workflow stages from CLI.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument(
        "--stages",
        default="all",
        help="Comma list among 1,2,3,4,5,6,7 or 'all'. Example: 1,2,3",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--stage2-variants",
        choices=["baseline", "all"],
        default="baseline",
        help="Stage 2 prep target variants.",
    )
    p.add_argument(
        "--stage2-splits",
        default="val",
        help="Comma list for Stage 2 prep splits (train,val,full). Default: val",
    )
    p.add_argument("--stage5-prepare-inputs", action="store_true", help="Prepare quantity variant token/feature/manifest inputs.")
    p.add_argument("--stage4-run-experiments", action="store_true", help="Run and store stage-4 variant experiments.")
    p.add_argument("--stage4-run-all-missing", action="store_true", help="Run all missing variants in stage-4 results store.")
    p.add_argument("--stage4-allow-overwrite-results", action="store_true", help="Allow overwriting existing stage-4 variant result entries.")
    p.add_argument("--stage5-run-experiments", action="store_true", help="Run quantity-variant experiments in stage-5.")
    p.add_argument("--stage5-allow-overwrite-results", action="store_true", help="Allow overwriting existing stage-5 quantity result entries.")
    p.add_argument("--experiment-epochs", type=int, default=1)
    p.add_argument("--experiment-batch-size", type=int, default=8)
    p.add_argument("--experiment-log-every-steps", type=int, default=20)
    p.add_argument("--experiment-learning-rate", type=float, default=2e-5)
    p.add_argument("--experiment-weight-decay", type=float, default=0.0)
    p.add_argument("--experiment-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--experiment-use-mesh", action="store_true")
    p.add_argument("--experiment-seed", type=int, default=0)
    p.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--run-name", default="group2_workflow")
    p.add_argument("--max-rows-guard", type=int, default=10000)
    p.add_argument("--subset-token", default="subset_10000_seed42")
    p.add_argument("--allow-non-subset", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(raw_cfg, PROJECT_ROOT)

    stage2_root = Path(cfg["stage2_root"])
    image_root = Path(cfg["image_root"])
    feature_root = Path(cfg["clip_feature_root"])
    reuse_feature_roots = [Path(p) for p in cfg.get("reuse_clip_feature_roots", [])]
    baseline_variant = str(cfg["baseline_variant"])
    quality_variants = [str(v) for v in cfg.get("quality_variants", [])]
    all_variants = [baseline_variant] + quality_variants
    results_path = stage2_root / "all_results_manual.json"

    if args.stages == "all":
        enabled = {str(i) for i in range(1, 8)}
    else:
        enabled = {s.strip() for s in args.stages.split(",") if s.strip()}

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("stage2_root:", stage2_root)
    print("all_variants:", all_variants)
    print("overwrite:", args.overwrite)
    if not args.allow_non_subset and args.subset_token not in str(stage2_root):
        raise RuntimeError(
            f"Refusing non-subset Group2 root: {stage2_root}. Expected token '{args.subset_token}'. "
            "Pass --allow-non-subset to override intentionally."
        )
    tracker = RunTracker(
        group="group2",
        output_root=Path(args.output_root),
        run_name=args.run_name,
        config={"args": vars(args), "project_root": str(PROJECT_ROOT), "config": str(config_path)},
    )
    print("RUN_DIR:", tracker.run_dir)

    runtime: dict[str, Any] | None = None

    def ensure_runtime() -> dict[str, Any]:
        nonlocal runtime
        if runtime is None:
            runtime = create_stage2_runtime_objects(cfg)
            print("runtime ready")
        return runtime

    def _runner(variant_name: str) -> dict[str, Any]:
        return run_stage2_experiment(
            project_root=PROJECT_ROOT,
            stage2_root=stage2_root,
            variant=variant_name,
            cfg=cfg,
            num_epochs=args.experiment_epochs,
            batch_size=args.experiment_batch_size,
            log_every_steps=args.experiment_log_every_steps,
            learning_rate=args.experiment_learning_rate,
            weight_decay=args.experiment_weight_decay,
            dtype=args.experiment_dtype,
            use_mesh=args.experiment_use_mesh,
            seed=args.experiment_seed,
        )

    # Stage 1
    if "1" in enabled:
        with tracker.stage("stage1_audit_split") as m:
            print("\n[Stage 1] Audit + shared split")
            audit = audit_stage2_variants(stage2_root, all_variants)
            row_counts = audit.get("variant_row_counts", {})
            print("audit variants:", list(row_counts.keys()))
            if args.max_rows_guard > 0:
                viol = {k: int(v) for k, v in row_counts.items() if int(v) > args.max_rows_guard}
                if viol:
                    raise RuntimeError(f"Group2 row-count guard failed: {viol} exceeds {args.max_rows_guard}")
            pool = build_shared_quality_pool(
                stage2_root=stage2_root,
                all_variants=all_variants,
                quality_image_count=int(cfg["quality_image_count"]),
                val_image_count=int(cfg["val_image_count"]),
                split_seed=int(cfg.get("split_seed", 42)),
                pool_reference_variant=baseline_variant,
                overwrite=args.overwrite,
            )
            print("pool write:", pool.get("pool_write"), "split write:", pool.get("split_write"))
            split = materialize_train_val_split(stage2_root, all_variants, overwrite=args.overwrite)
            print("split variants:", list(split.keys()))
            m.update({"variants": len(all_variants), "split_variants": len(split)})

    if "2" in enabled:
        with tracker.stage("stage2_variant_prep") as m:
            print("\n[Stage 2] Variant token/feature/manifest prep")
            rt = ensure_runtime()
            target_variants = [baseline_variant] if args.stage2_variants == "baseline" else all_variants
            splits = tuple(s.strip() for s in args.stage2_splits.split(",") if s.strip())
            prep = prepare_stage2_variant_splits(
                stage2_root=stage2_root,
                image_root=image_root,
                feature_root=feature_root,
                tokenizer=rt["tokenizer"],
                clip_bundle=rt["clip_bundle"],
                get_features_compiled=rt["get_features_compiled"],
                all_variants=target_variants,
                splits=splits,
                overwrite=args.overwrite,
                additional_feature_roots=reuse_feature_roots,
            )
            print("prepared jobs:", len(prep))
            print("sample:", prep[0] if prep else "<none>")
            m.update({"prepared_jobs": len(prep), "target_variants": len(target_variants), "splits": ",".join(splits)})

    if "3" in enabled:
        with tracker.stage("stage3_eval_artifacts") as m:
            print("\n[Stage 3] Evaluation artifacts")
            diag = build_dataset_quality_diagnostics(stage2_root, all_variants, overwrite=args.overwrite)
            qual = build_qualitative_samples_pack(stage2_root, all_variants, overwrite=args.overwrite)
            print("quality diagnostics:", diag.get("mode", "generated"))
            print("qual samples:", qual.get("mode", "generated"))
            m.update({"diag_mode": diag.get("mode", "generated"), "qual_mode": qual.get("mode", "generated")})

    if "4" in enabled:
        with tracker.stage("stage4_tracking_summary") as m:
            print("\n[Stage 4] Experiment tracking summaries")
            selection = select_next_variant(
                results_path=results_path,
                expected_variants=all_variants,
                current_variant=None,
                allow_overwrite=args.stage4_allow_overwrite_results,
            )
            print("missing_variants:", selection["missing_variants"])
            pa = prompt_alignment_audit(stage2_root, all_variants, prompt_reference_variant=baseline_variant, overwrite=args.overwrite)
            print("prompt audit:", pa.get("mode", "generated"))
            m.update({"missing_variants": len(selection["missing_variants"]), "prompt_audit_mode": pa.get("mode", "generated")})

            if args.stage4_run_experiments:
                to_run: list[str]
                if args.stage4_run_all_missing:
                    to_run = list(selection["missing_variants"])
                else:
                    to_run = [selection["selected_variant"]] if selection["selected_variant"] is not None else []
                print("stage4 experiment variants:", to_run if to_run else "<none>")

                all_results = selection["all_results"]
                for variant in to_run:
                    run_out = run_and_store_variant(
                        results_path=results_path,
                        all_results=all_results,
                        selected_variant=variant,
                        run_stage2_experiment_fn=_runner,
                    )
                    all_results = run_out["all_results"]
                    print("stored variant result:", run_out["ran_variant"])
                # refresh selection after potential new writes
                selection = select_next_variant(
                    results_path=results_path,
                    expected_variants=all_variants,
                    current_variant=None,
                    allow_overwrite=args.stage4_allow_overwrite_results,
                )
                print("missing_variants_after_run:", selection["missing_variants"])

            if not results_path.exists():
                print("summary blocked: missing", results_path)
            else:
                all_results = json.loads(results_path.read_text(encoding="utf-8"))
                missing = [v for v in all_variants if v not in all_results]
                if missing:
                    print("summary blocked: missing variants in results file:", missing)
                else:
                    es = build_engine_comparison_summary(stage2_root, results_path, all_variants, overwrite=args.overwrite)
                    br = build_baseline_relative_comparison(
                        stage2_root, results_path, baseline_variant=baseline_variant, expected_variants=all_variants, overwrite=args.overwrite
                    )
                    print("engine summary:", es.get("mode", "generated"))
                    print("baseline relative:", br.get("mode", "generated"))
                    m.update({"engine_summary_mode": es.get("mode", "generated"), "baseline_relative_mode": br.get("mode", "generated")})

    if "5" in enabled:
        with tracker.stage("stage5_quantity_ablation") as m:
            print("\n[Stage 5] Quantity ablation")
            engine_summary = stage2_root / "engine_comparison_summary.json"
            if not engine_summary.exists():
                print("blocked: missing", engine_summary)
            else:
                plan = derive_quantity_plan(stage2_root)
                qvars = build_quantity_variants(
                    stage2_root=stage2_root,
                    quantity_source_variant=plan["quantity_source_variant"],
                    quantity_levels=plan["quantity_levels"],
                    quantity_split_seed=plan["quantity_split_seed"],
                    overwrite=args.overwrite,
                )
                reg = register_quantity_variants(stage2_root, qvars, overwrite=args.overwrite)
                print("quantity variants:", qvars)
                print("registered:", len(reg))
                m.update({"quantity_variants": len(qvars), "registered": len(reg)})

                if args.stage5_prepare_inputs:
                    rt = ensure_runtime()

                    def tok_cb(variant: str, split: str, overwrite: bool = False) -> dict:
                        return tokenize_stage2_variant(stage2_root, rt["tokenizer"], variant, split, overwrite=overwrite)

                    def feat_cb(variant: str, split: str, overwrite: bool = False) -> dict:
                        return extract_stage2_features(
                            stage2_root,
                            image_root,
                            feature_root,
                            rt["clip_bundle"],
                            rt["get_features_compiled"],
                            variant,
                            split,
                            overwrite=overwrite,
                            additional_feature_roots=reuse_feature_roots,
                        )

                    def man_cb(variant: str, split: str, overwrite: bool = False) -> dict:
                        return build_stage2_manifest(
                            stage2_root,
                            feature_root,
                            variant,
                            split,
                            overwrite=overwrite,
                            additional_feature_roots=reuse_feature_roots,
                        )

                    prep = prepare_quantity_variants(
                        stage2_root=stage2_root,
                        quantity_variants=qvars,
                        tokenize_fn=tok_cb,
                        extract_features_fn=feat_cb,
                        build_manifest_fn=man_cb,
                        overwrite=args.overwrite,
                    )
                    print("quantity prep entries:", len(prep))
                    m.update({"quantity_prep_entries": len(prep)})

                if args.stage5_run_experiments:
                    qresults = run_quantity_experiments(
                        stage2_root=stage2_root,
                        quantity_variants=qvars,
                        run_stage2_experiment_fn=_runner,
                        allow_overwrite=args.stage5_allow_overwrite_results,
                    )
                    print("quantity experiment results:", len(qresults))
                    m.update({"quantity_results_entries": len(qresults)})

                qres = stage2_root / "quantity_results.json"
                if qres.exists():
                    qsum = summarize_quantity_results(stage2_root)
                    print("quantity summary rows:", len(qsum.get("quantity_ranking", [])))
                    m.update({"quantity_summary_rows": len(qsum.get("quantity_ranking", []))})
                else:
                    print("quantity summary skipped: quantity_results.json missing")

    if "6" in enabled:
        with tracker.stage("stage6_heldout_pairwise") as m:
            print("\n[Stage 6] Heldout pack + pairwise requests")
            heldout = build_heldout_eval_pack(stage2_root, all_variants, overwrite=args.overwrite)
            pairwise = build_pairwise_judge_requests(stage2_root, baseline_variant=baseline_variant, overwrite=args.overwrite)
            print("heldout:", heldout.get("mode", "generated"))
            print("pairwise:", pairwise.get("mode", "generated"))
            m.update({"heldout_mode": heldout.get("mode", "generated"), "pairwise_mode": pairwise.get("mode", "generated")})

    if "7" in enabled:
        with tracker.stage("stage7_reporting") as m:
            print("\n[Stage 7] Reporting figures")
            required = [
                stage2_root / "engine_comparison_summary.json",
                stage2_root / "dataset_quality_diagnostics.json",
            ]
            missing = [p for p in required if not p.exists()]
            if missing:
                print("reporting blocked, missing:")
                for p in missing:
                    print(" -", p)
                m.update({"missing_required": len(missing)})
            else:
                out = build_engine_plots_and_table(stage2_root, overwrite=args.overwrite)
                print("reporting:", out.get("mode", "generated"))
                print("table:", out.get("summary_table_md"))
                m.update({"reporting_mode": out.get("mode", "generated")})

    summary = tracker.finalize()
    print("Run summary:", summary["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
