from __future__ import annotations

import json
import math
from pathlib import Path


def build_engine_plots_and_table(stage2_root: Path) -> dict:
    import matplotlib.pyplot as plt

    engine_summary = json.loads((stage2_root / "engine_comparison_summary.json").read_text(encoding="utf-8"))
    quality_diag = json.loads((stage2_root / "dataset_quality_diagnostics.json").read_text(encoding="utf-8"))
    ranking = engine_summary["variant_summaries"]
    diag_map = {row["variant"]: row for row in quality_diag["diagnostics"]}

    variants = [row["variant"] for row in ranking]
    final_train_losses = [row["final_train_mean_loss"] for row in ranking]
    final_val_losses = [row["final_val_mean_loss"] for row in ranking]
    mean_response_words = [diag_map[v]["overall"]["mean_response_words"] if v in diag_map else None for v in variants]

    plots_dir = stage2_root / "report_figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def clean_values(values):
        return [float(v) if v is not None else math.nan for v in values]

    plt.figure(figsize=(8, 5))
    plt.bar(variants, clean_values(final_val_losses))
    plt.title("Final Validation Loss by Engine")
    plt.ylabel("Mean validation loss")
    plt.xlabel("Engine variant")
    plt.xticks(rotation=20)
    plt.tight_layout()
    val_plot_path = plots_dir / "final_validation_loss.png"
    plt.savefig(val_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(variants, clean_values(final_train_losses))
    plt.title("Final Training Loss by Engine")
    plt.ylabel("Mean training loss")
    plt.xlabel("Engine variant")
    plt.xticks(rotation=20)
    plt.tight_layout()
    train_plot_path = plots_dir / "final_training_loss.png"
    plt.savefig(train_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(variants, clean_values(mean_response_words))
    plt.title("Average Response Length by Engine")
    plt.ylabel("Mean response words")
    plt.xlabel("Engine variant")
    plt.xticks(rotation=20)
    plt.tight_layout()
    resp_plot_path = plots_dir / "mean_response_words.png"
    plt.savefig(resp_plot_path, dpi=200)
    plt.close()

    summary_md_path = plots_dir / "engine_results_table.md"
    with summary_md_path.open("w", encoding="utf-8") as f:
        f.write("# Engine Results Summary\n\n")
        f.write("| Variant | Final Train Loss | Final Val Loss | Mean Response Words |\n")
        f.write("|---|---:|---:|---:|\n")
        for v, tr, va, rw in zip(variants, final_train_losses, final_val_losses, mean_response_words):
            tr_str = f"{tr:.4f}" if tr is not None else "NA"
            va_str = f"{va:.4f}" if va is not None else "NA"
            rw_str = f"{rw:.2f}" if rw is not None else "NA"
            f.write(f"| {v} | {tr_str} | {va_str} | {rw_str} |\n")

    return {
        "final_validation_plot": str(val_plot_path),
        "final_training_plot": str(train_plot_path),
        "mean_response_words_plot": str(resp_plot_path),
        "summary_table_md": str(summary_md_path),
    }

