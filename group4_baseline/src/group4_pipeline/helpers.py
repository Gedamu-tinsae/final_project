"""Reusable helpers for Group4 pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def expand_project_root(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    token = "${PROJECT_ROOT}"
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        elif isinstance(v, list):
            out[k] = [x.replace(token, str(project_root)) if isinstance(x, str) else x for x in v]
        elif isinstance(v, dict):
            out[k] = expand_project_root(v, project_root)
        else:
            out[k] = v
    return out


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_plot_series(out_png: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not xs or not ys:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def default_group4_config(project_root: Path) -> dict[str, Any]:
    return {
        "project_root": str(project_root),
        "group1_root": str(project_root / "../group1_baseline"),
        "group2_root": str(project_root / "../group2_baseline"),
        "required_inputs": {
            "group1_stage1_manifest": str(project_root / "../group1_baseline/data/processed/subsets/subset_10000_seed42/stage1_alignment/stage1_manifest.json"),
            "group1_stage2_manifest": str(project_root / "../group1_baseline/data/processed/subsets/subset_10000_seed42/stage2_finetuning/stage2_manifest.json"),
            "group1_stage1_projector_state": str(project_root / "../group1_baseline/artifacts/subsets/subset_10000_seed42/projector_stage1.pkl"),
            "group1_stage2_projector_state": str(project_root / "../group1_baseline/artifacts/subsets/subset_10000_seed42/projector_stage2.pkl"),
            "group1_stage2_llama_state": str(project_root / "../group1_baseline/artifacts/subsets/subset_10000_seed42/llama_stage2.pkl"),
            "group2_engine_summary": str(project_root / "../group2_baseline/data/processed/subsets/subset_10000_seed42/stage2_instruction/engine_comparison_summary.json"),
        },
        "group4_outputs": {
            "plan_json": str(project_root / "data/processed/subsets/subset_10000_seed42/group4_experiment_plan.json"),
            "run_registry_json": str(project_root / "data/processed/subsets/subset_10000_seed42/group4_run_registry.json"),
            "results_manual_json": str(project_root / "data/processed/subsets/subset_10000_seed42/group4_results_manual.json"),
            "summary_json": str(project_root / "data/processed/subsets/subset_10000_seed42/group4_results_summary.json"),
            "summary_md": str(project_root / "data/processed/subsets/subset_10000_seed42/group4_results_summary.md"),
        },
        "experiment_space": {
            "methods": ["lora", "selective_ft"],
            "lora_ranks": [4, 8, 16],
            "target_modules": ["qv", "all"],
            "selective_ft_budget_pct": [0.1, 0.5, 1.0],
            "train_budget_steps": 500,
            "seed": 42,
        },
    }


def resolve_group4_config(project_root: Path, config_arg: str) -> tuple[dict[str, Any], Path | None]:
    candidate_paths = [
        (project_root / config_arg).resolve(),
        (project_root / "configs/workflow_paths_subset_10000.json").resolve(),
        (project_root / "configs/workflow_paths.json").resolve(),
    ]
    seen: set[Path] = set()
    for p in candidate_paths:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return expand_project_root(raw, project_root), p
    # Final fallback: built-in defaults.
    return default_group4_config(project_root), None
