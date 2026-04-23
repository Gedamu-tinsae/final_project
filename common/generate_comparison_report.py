"""Build cross-run comparison tables/figures from outputs/ directories."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate cross-run comparison artifacts.")
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--run-name", default="comparison")
    return p.parse_args()


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Comparison Table\n\nNo rows found.\n", encoding="utf-8")
        return
    cols = ["group", "run_id", "wall_time_sec", "rss_kb_max", "method", "val_loss", "trainable_params_millions", "steps_per_sec"]
    lines = ["# Comparison Table", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    if not labels or not values:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _try_dashboard(path: Path, images: list[Path]) -> None:
    imgs = [p for p in images if p.exists()]
    if not imgs:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.image as mpimg  # type: ignore
    except Exception:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(4):
        ax = axes[i]
        if i < len(imgs):
            ax.imshow(mpimg.imread(imgs[i]))
            ax.set_title(imgs[i].name)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _collect_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_dir in outputs_root.glob("*"):
        if not group_dir.is_dir() or group_dir.name == "comparison":
            continue
        for run_dir in group_dir.glob("*"):
            if not run_dir.is_dir():
                continue
            ms = _read_json(run_dir / "metrics_summary.json")
            if ms is None:
                continue
            row = {
                "group": ms.get("group", group_dir.name),
                "run_id": ms.get("run_id", run_dir.name),
                "wall_time_sec": ms.get("wall_time_sec"),
                "rss_kb_max": (ms.get("resources") or {}).get("rss_kb_max"),
                "method": "",
                "val_loss": "",
                "trainable_params_millions": "",
                "steps_per_sec": "",
            }
            rows.append(row)

            peft = _read_json(run_dir / "peft" / "metrics.json")
            if peft is not None:
                rows.append(
                    {
                        "group": "group4_peft",
                        "run_id": peft.get("run_id", run_dir.name),
                        "wall_time_sec": peft.get("wall_time_sec"),
                        "rss_kb_max": (ms.get("resources") or {}).get("rss_kb_max"),
                        "method": peft.get("method", ""),
                        "val_loss": peft.get("val_loss", ""),
                        "trainable_params_millions": peft.get("trainable_params_millions", ""),
                        "steps_per_sec": peft.get("steps_per_sec", ""),
                    }
                )
    return rows


def _write_report_index(outputs_root: Path, comparison_dir: Path) -> None:
    lines = ["# Report Index", "", "## Comparison Runs", ""]
    for d in sorted((outputs_root / "comparison").glob("*"), reverse=True):
        if d.is_dir():
            lines.append(f"- {d.name}: {d}")
    lines += ["", "## Group Runs", ""]
    for g in ["group1", "group2", "group4"]:
        gd = outputs_root / g
        if not gd.exists():
            continue
        lines.append(f"### {g}")
        for d in sorted(gd.glob("*"), reverse=True)[:10]:
            if d.is_dir():
                lines.append(f"- {d.name}: {d}")
        lines.append("")
    (outputs_root / "REPORT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    run_id = f"{_ts()}_{args.run_name}"
    out_dir = outputs_root / "comparison" / run_id
    figs = out_dir / "comparison_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(outputs_root)
    _write_csv(out_dir / "comparison_table.csv", rows)
    _write_md(out_dir / "comparison_table.md", rows)

    # Group comparison (wall time)
    group_wall: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for r in rows:
        g = str(r.get("group", ""))
        try:
            w = float(r.get("wall_time_sec") or 0.0)
        except Exception:
            w = 0.0
        group_wall[g] = group_wall.get(g, 0.0) + w
        group_counts[g] = group_counts.get(g, 0) + 1
    labels = []
    vals = []
    for g, total in group_wall.items():
        if g:
            labels.append(g)
            vals.append(total / max(1, group_counts.get(g, 1)))
    _try_plot_bar(figs / "fig_group_comparison.png", labels, vals, "Average Wall Time by Group", "seconds")

    # Method comparison (group4 peft)
    peft_rows = [r for r in rows if str(r.get("group")) == "group4_peft"]
    method_labels = []
    method_vals = []
    trainable_labels = []
    trainable_vals = []
    for r in peft_rows:
        method = str(r.get("method", ""))
        if not method:
            continue
        method_labels.append(method)
        try:
            method_vals.append(float(r.get("val_loss") or 0.0))
        except Exception:
            method_vals.append(0.0)
        trainable_labels.append(method)
        try:
            trainable_vals.append(float(r.get("trainable_params_millions") or 0.0))
        except Exception:
            trainable_vals.append(0.0)

    _try_plot_bar(
        figs / "fig_method_comparison_bar.png",
        method_labels,
        method_vals,
        "Method Comparison (Val Loss)",
        "val_loss",
    )
    _try_plot_bar(
        figs / "fig_methods_overview.png",
        method_labels,
        method_vals,
        "Methods Overview (Val Loss)",
        "val_loss",
    )
    _try_plot_bar(
        figs / "fig_trainable_params_bar.png",
        trainable_labels,
        trainable_vals,
        "Trainable Params (Millions)",
        "M params",
    )

    _try_dashboard(
        figs / "fig_presentation_dashboard.png",
        [
            figs / "fig_group_comparison.png",
            figs / "fig_method_comparison_bar.png",
            figs / "fig_trainable_params_bar.png",
            figs / "fig_methods_overview.png",
        ],
    )

    # Aliases required by plan wording
    if (figs / "fig_presentation_dashboard.png").exists():
        (figs / "fig_overview_dashboard.png").write_bytes((figs / "fig_presentation_dashboard.png").read_bytes())

    _write_report_index(outputs_root, out_dir)

    print("comparison_dir:", out_dir)
    print("comparison_table:", out_dir / "comparison_table.csv")
    print("report_index:", outputs_root / "REPORT_INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
