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
            "methods": ["lora", "selective_ft", "relora"],
            "lora_ranks": [4, 8, 16],
            "target_modules": ["qv", "all"],
            "selective_ft_budget_pct": [0.1, 0.5, 1.0],
            "relora_merge_freq": [500],
            "relora_final_merge": True,
            "train_budget_steps": 500,
            "seed": 42,
        },
    }


def resolve_group4_config(project_root: Path, config_arg: str) -> tuple[dict[str, Any], Path | None]:
    requested = (project_root / config_arg).resolve()
    default_subset = (project_root / "configs/workflow_paths_subset_10000.json").resolve()
    default_base = (project_root / "configs/workflow_paths.json").resolve()

    # Safety: if caller explicitly points to a non-default config path and it does
    # not exist, fail fast. Do not silently fall back to another config.
    default_tokens = {"configs/workflow_paths_subset_10000.json", "configs/workflow_paths.json"}
    if config_arg not in default_tokens and not requested.exists():
        raise FileNotFoundError(f"Requested config not found: {requested}")

    candidate_paths = [
        requested,
        default_subset,
        default_base,
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


def normalize_experiment_space(exp: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize/validate experiment space for robust Group4 staging.

    Returns normalized config plus warning messages for migrated legacy fields.
    """
    warnings: list[str] = []

    methods_raw = [str(x) for x in exp.get("methods", ["lora", "selective_ft", "relora"])]
    methods: list[str] = []
    for m in methods_raw:
        mm = m.strip().lower()
        if mm not in {"lora", "selective_ft", "relora"}:
            raise ValueError(f"Unsupported method in experiment_space.methods: {m}")
        if mm not in methods:
            methods.append(mm)

    targets_raw = [str(x) for x in exp.get("target_modules", ["qv", "all"])]
    targets: list[str] = []
    for t in targets_raw:
        tt = t.strip().lower()
        if tt == "qv_mlp":
            tt = "all"
            warnings.append("Mapped legacy target_modules value 'qv_mlp' to 'all'.")
        if tt not in {"qv", "all"}:
            raise ValueError(f"Unsupported target module in experiment_space.target_modules: {t}")
        if tt not in targets:
            targets.append(tt)

    ranks_raw = exp.get("lora_ranks", [4, 8, 16])
    lora_ranks: list[int] = []
    for r in ranks_raw:
        rr = int(r)
        if rr <= 0:
            raise ValueError(f"Invalid LoRA rank: {rr}")
        if rr not in lora_ranks:
            lora_ranks.append(rr)

    budgets_raw = exp.get("selective_ft_budget_pct", [0.1, 0.5, 1.0])
    selective_budgets: list[float] = []
    for b in budgets_raw:
        bb = float(b)
        if bb <= 0.0:
            raise ValueError(f"Invalid selective_ft budget pct: {bb}")
        if bb not in selective_budgets:
            selective_budgets.append(bb)

    train_budget_steps = int(exp.get("train_budget_steps", 500))
    if train_budget_steps <= 0:
        raise ValueError(f"Invalid train_budget_steps: {train_budget_steps}")

    seed = int(exp.get("seed", 42))
    relora_merge_freq_raw = exp.get("relora_merge_freq", [500])
    relora_merge_freq: list[int] = []
    for f in relora_merge_freq_raw:
        ff = int(f)
        if ff <= 0:
            raise ValueError(f"Invalid relora merge frequency: {ff}")
        if ff not in relora_merge_freq:
            relora_merge_freq.append(ff)
    relora_final_merge = bool(exp.get("relora_final_merge", True))

    normalized = {
        "methods": methods,
        "lora_ranks": lora_ranks,
        "target_modules": targets,
        "selective_ft_budget_pct": selective_budgets,
        "relora_merge_freq": relora_merge_freq,
        "relora_final_merge": relora_final_merge,
        "train_budget_steps": train_budget_steps,
        "seed": seed,
    }
    return normalized, warnings
