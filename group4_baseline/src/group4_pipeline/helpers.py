"""Reusable helpers for Group4 pipeline."""

from __future__ import annotations

import importlib.util
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
