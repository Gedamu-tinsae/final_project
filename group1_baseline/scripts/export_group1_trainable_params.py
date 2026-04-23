#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import load_dotenv_file, load_json_config

# Param-audit is read-only and should not contend with active TPU training jobs.
os.environ["JAX_PLATFORMS"] = "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Group1 trainable parameter audit (read-only).")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "param_audit"))
    p.add_argument("--run-name", default="group1_param_audit")
    return p.parse_args()


def _safe_jax_backend() -> str:
    try:
        import jax  # type: ignore

        return str(jax.default_backend())
    except Exception as e:
        return f"unavailable:{type(e).__name__}"


def _safe_jax_num_devices() -> int:
    try:
        import jax  # type: ignore

        return int(len(jax.devices()))
    except Exception:
        return 0


def _count_params(tree: Any) -> int:
    try:
        import jax  # type: ignore

        leaves = jax.tree_util.tree_leaves(tree)
    except Exception:
        # fallback for non-jax environments
        leaves = []
        if isinstance(tree, dict):
            stack = [tree]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
                else:
                    leaves.append(item)

    total = 0
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape is None:
            continue
        n = 1
        for d in shape:
            n *= int(d)
        total += int(n)
    return int(total)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| stage | projector_total_params | projector_trainable_params | llama_total_params | llama_trainable_params | total_trainable_params | trainable_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {stage} | {projector_total_params} | {projector_trainable_params} | {llama_total_params} | {llama_trainable_params} | {total_trainable_params} | {trainable_ratio:.6f} |".format(
                **r
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_plot_trainable_bar(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    stages = [str(r["stage"]) for r in rows]
    values_m = [float(r["total_trainable_params"]) / 1_000_000.0 for r in rows]
    plt.figure(figsize=(6, 4))
    plt.bar(stages, values_m)
    plt.ylabel("Trainable Params (Millions)")
    plt.title("Group1 Trainable Parameters by Stage")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv_file(dotenv_path)
    cfg = load_json_config(config_path, PROJECT_ROOT)

    output_root = Path(args.output_root).resolve()
    allowed_root = (PROJECT_ROOT / "outputs" / "param_audit").resolve()
    if output_root != allowed_root:
        raise RuntimeError(
            f"Refusing output_root outside dedicated param audit dir. "
            f"Expected exactly: {allowed_root} ; got: {output_root}"
        )
    run_id = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{args.run_name}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stage1_projector_state = Path(cfg["stage1_projector_state"])
    stage2_projector_state = Path(cfg["stage2_projector_state"])
    stage2_llama_state = Path(cfg["stage2_llama_state"])
    llama_local_dir = Path(cfg.get("llama_local_dir", PROJECT_ROOT / "data" / "models" / "Llama-3.2-1B-Instruct"))

    required = {
        "stage1_projector_state": stage1_projector_state,
        "stage2_projector_state": stage2_projector_state,
        "stage2_llama_state": stage2_llama_state,
        "llama_local_dir": llama_local_dir,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required read-only artifacts: {missing}")

    with stage1_projector_state.open("rb") as f:
        projector_stage1 = pickle.load(f)
    with stage2_projector_state.open("rb") as f:
        projector_stage2 = pickle.load(f)
    with stage2_llama_state.open("rb") as f:
        llama_stage2 = pickle.load(f)

    projector_total_params = _count_params(projector_stage1)
    projector_stage2_total_params = _count_params(projector_stage2)
    llama_total_params = _count_params(llama_stage2)

    row_stage1 = {
        "stage": "stage1",
        "projector_total_params": int(projector_total_params),
        "projector_trainable_params": int(projector_total_params),
        "llama_total_params": int(llama_total_params),
        "llama_trainable_params": 0,
        "total_trainable_params": int(projector_total_params),
        "trainable_ratio": float(projector_total_params / max(1, projector_total_params + llama_total_params)),
    }
    row_stage2 = {
        "stage": "stage2",
        "projector_total_params": int(projector_stage2_total_params),
        "projector_trainable_params": int(projector_stage2_total_params),
        "llama_total_params": int(llama_total_params),
        "llama_trainable_params": int(llama_total_params),
        "total_trainable_params": int(projector_stage2_total_params + llama_total_params),
        "trainable_ratio": float((projector_stage2_total_params + llama_total_params) / max(1, projector_stage2_total_params + llama_total_params)),
    }
    rows = [row_stage1, row_stage2]

    summary = {
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "artifacts": {k: str(v) for k, v in required.items()},
        "schema": {
            "projector_total_params": row_stage1["projector_total_params"],
            "projector_trainable_params": row_stage1["projector_trainable_params"],
            "llama_total_params": row_stage2["llama_total_params"],
            "llama_trainable_params_stage1": row_stage1["llama_trainable_params"],
            "llama_trainable_params_stage2": row_stage2["llama_trainable_params"],
            "total_trainable_stage1": row_stage1["total_trainable_params"],
            "total_trainable_stage2": row_stage2["total_trainable_params"],
            "trainable_ratio_stage1": row_stage1["trainable_ratio"],
            "trainable_ratio_stage2": row_stage2["trainable_ratio"],
        },
    }

    _write_json(
        run_dir / "run_config.json",
        {"args": vars(args), "config": str(config_path), "run_id": run_id},
    )
    _write_json(
        run_dir / "system_info.json",
        {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
            "jax_backend": _safe_jax_backend(),
            "jax_num_devices": _safe_jax_num_devices(),
        },
    )
    _write_json(run_dir / "trainable_params_summary.json", summary)
    _write_csv(run_dir / "trainable_params_table.csv", rows)
    _write_md_table(run_dir / "trainable_params_table.md", rows)
    has_fig = _try_plot_trainable_bar(run_dir / "fig_trainable_params_bar.png", rows)
    _write_json(run_dir / "artifacts_manifest.json", {"plot_generated": bool(has_fig)})

    print("PARAM_AUDIT_RUN_DIR:", run_dir)
    print("projector_total_params:", row_stage1["projector_total_params"])
    print("llama_total_params:", row_stage2["llama_total_params"])
    print("total_trainable_stage1:", row_stage1["total_trainable_params"])
    print("total_trainable_stage2:", row_stage2["total_trainable_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
