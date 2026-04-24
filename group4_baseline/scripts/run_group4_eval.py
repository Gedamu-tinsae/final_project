"""Run Group4 modular evaluation (human and API judging)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.group4_pipeline.eval import (
    aggregate_pairwise_results,
    build_generations_template_from_manifest,
    build_pairwise_requests,
    run_openai_judging,
    update_group4_results_with_eval,
    write_human_eval_pack,
)
from src.group4_pipeline.helpers import expand_project_root
from src.group4_pipeline.helpers import resolve_group4_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group4 evaluation runner (template/human/API).")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--generations-jsonl", default="", help="JSONL with fields: sample_id,prompt,method,output")
    p.add_argument("--manifest-json", default="", help="Manifest JSON list used for template generation.")
    p.add_argument("--baseline-method", default="baseline")
    p.add_argument("--candidate-methods", default="", help="Comma list; default uses all non-baseline methods.")
    p.add_argument("--template-methods", default="", help="Comma list for template mode.")
    p.add_argument("--template-max-samples", type=int, default=200)
    p.add_argument("--output-dir", default="", help="Default: <group4_outputs parent>/eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--mode", choices=["template", "human_pack", "human_aggregate", "api_judge"], default="human_pack")
    p.add_argument("--human-results-jsonl", default="", help="Required for mode=human_aggregate")
    p.add_argument("--openai-model", default="gpt-4.1-mini")
    p.add_argument("--openai-api-key", default="")
    p.add_argument("--update-results-manual", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg, cfg_path = resolve_group4_config(PROJECT_ROOT, args.config)

    out_dir = Path(args.output_dir) if args.output_dir else (Path(cfg["group4_outputs"]["summary_json"]).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_manual_json = Path(cfg["group4_outputs"]["results_manual_json"])

    if args.mode == "template":
        manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else Path(cfg["required_inputs"]["group1_stage1_manifest"]).resolve()
        if not manifest_json.exists():
            raise FileNotFoundError(manifest_json)
        if args.template_methods.strip():
            methods = [s.strip() for s in args.template_methods.split(",") if s.strip()]
        else:
            methods = ["baseline"]
            if results_manual_json.exists():
                data = json.loads(results_manual_json.read_text(encoding="utf-8"))
                for r in data.get("results", []):
                    m = str(r.get("experiment_id", "")).strip()
                    if m:
                        methods.append(m)
            methods = list(dict.fromkeys(methods))
        out_template = out_dir / "group4_generations_template.jsonl"
        meta = build_generations_template_from_manifest(
            manifest_json,
            methods=methods,
            out_jsonl=out_template,
            max_samples=int(args.template_max_samples),
        )
        (out_dir / "group4_generations_template.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("template:", meta)
        return 0

    if not args.generations_jsonl:
        raise ValueError("--generations-jsonl is required for this mode.")
    generations_jsonl = Path(args.generations_jsonl).resolve()
    if not generations_jsonl.exists():
        raise FileNotFoundError(generations_jsonl)

    candidates = [s.strip() for s in args.candidate_methods.split(",") if s.strip()]
    pairwise = build_pairwise_requests(
        generations_jsonl,
        baseline_method=args.baseline_method,
        candidate_methods=candidates or None,
        seed=args.seed,
        max_requests=args.max_requests,
    )
    pairwise_meta_path = out_dir / "pairwise_meta.json"
    pairwise_requests_path = out_dir / "pairwise_requests.jsonl"
    pairwise_meta_path.write_text(json.dumps({k: v for k, v in pairwise.items() if k != "requests"}, ensure_ascii=False, indent=2), encoding="utf-8")
    with pairwise_requests_path.open("w", encoding="utf-8") as f:
        for r in pairwise["requests"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.mode == "human_pack":
        pack = write_human_eval_pack(out_dir, pairwise)
        print("human_pack:", pack)
        return 0

    if args.mode == "human_aggregate":
        if not args.human_results_jsonl:
            raise ValueError("--human-results-jsonl is required for mode=human_aggregate")
        summary = aggregate_pairwise_results(pairwise_requests_path, Path(args.human_results_jsonl).resolve())
        summary_path = out_dir / "human_eval_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("human_summary:", summary_path)
        if args.update_results_manual:
            upd = update_group4_results_with_eval(results_manual_json, summary)
            print("updated_results_manual:", upd)
        return 0

    # mode == api_judge
    judged = run_openai_judging(
        pairwise_requests_path,
        model=args.openai_model,
        api_key=args.openai_api_key or None,
        max_requests=args.max_requests,
    )
    judged_path = out_dir / "api_judged_results.jsonl"
    with judged_path.open("w", encoding="utf-8") as f:
        for r in judged["judged_rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = aggregate_pairwise_results(pairwise_requests_path, judged_path)
    summary_path = out_dir / "api_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("api_judged:", judged_path)
    print("api_summary:", summary_path)
    if args.update_results_manual:
        upd = update_group4_results_with_eval(results_manual_json, summary)
        print("updated_results_manual:", upd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
