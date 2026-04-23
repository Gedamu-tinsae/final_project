"""Create a deterministic subset profile for Group1 workflow.

This script:
1) Ensures full Stage-1 alignment/chat exist (generates if missing).
2) Builds a fixed-size subset from Stage-1 rows.
3) Writes subset Stage-1/Stage-2 input JSONs.
4) Writes a workflow config that points all downstream outputs to subset paths.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_json_config
from src.data_prep.stage1_pipeline import run_stage1_data_prep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create deterministic subset profile for Group1.")
    p.add_argument("--source-config", default="configs/workflow_paths.json")
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--profile-name",
        default="subset_10000_seed42",
        help="Output profile directory name under data/processed/subsets and artifacts/subsets.",
    )
    p.add_argument(
        "--output-config",
        default="configs/workflow_paths_subset_10000.json",
        help="Path (relative to project root) for generated workflow config.",
    )
    p.add_argument("--download", action="store_true", help="Allow COCO download if missing.")
    p.add_argument("--extract", action="store_true", help="Allow COCO extraction if missing.")
    return p.parse_args()


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_config = (PROJECT_ROOT / args.source_config).resolve()
    cfg = load_json_config(source_config, PROJECT_ROOT)

    full_alignment_path = Path(cfg["stage1_alignment_json"])
    full_chat_path = Path(cfg["stage1_chat_json"])

    # Ensure full stage-1 files exist first; subset is sampled from these.
    run_stage1_data_prep(
        project_root=PROJECT_ROOT,
        coco_json_path=Path(cfg["coco_json"]),
        alignment_path=full_alignment_path,
        chat_path=full_chat_path,
        seed=args.seed,
        overwrite=False,
        download=args.download,
        extract=args.extract,
    )

    alignment = _read_json(full_alignment_path)
    chat_rows = _read_json(full_chat_path)
    n = min(len(alignment), len(chat_rows))
    if n == 0:
        raise ValueError("Source stage1 files are empty; cannot build subset.")

    k = min(args.rows, n)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(n), k))

    alignment_subset = [alignment[i] for i in indices]
    chat_subset = [chat_rows[i] for i in indices]

    subset_processed_root = PROJECT_ROOT / "data" / "processed" / "subsets" / args.profile_name
    subset_artifacts_root = PROJECT_ROOT / "artifacts" / "subsets" / args.profile_name

    stage1_alignment_json = subset_processed_root / "stage1_alignment" / "alignment.json"
    stage1_chat_json = subset_processed_root / "stage1_alignment" / "alignment_chat.json"
    stage2_input_json = subset_processed_root / "stage2_finetuning" / "stage2_input.json"

    _write_json(stage1_alignment_json, alignment_subset)
    _write_json(stage1_chat_json, chat_subset)
    _write_json(stage2_input_json, chat_subset)

    generated_cfg = dict(cfg)
    generated_cfg.update(
        {
            "stage1_alignment_json": str(stage1_alignment_json),
            "stage1_chat_json": str(stage1_chat_json),
            "stage1_tokenized_json": str(
                subset_processed_root / "stage1_alignment" / "alignment_tokenized.json"
            ),
            "stage1_manifest_json": str(
                subset_processed_root / "stage1_alignment" / "stage1_manifest.json"
            ),
            "stage2_input_json": str(stage2_input_json),
            "stage2_tokenized_json": str(
                subset_processed_root / "stage2_finetuning" / "alignment_tokenized_stage2.json"
            ),
            "stage2_manifest_json": str(
                subset_processed_root / "stage2_finetuning" / "stage2_manifest.json"
            ),
            "stage2_manifest_smoke_json": str(
                subset_processed_root / "stage2_finetuning" / "stage2_manifest_smoke.json"
            ),
            "clip_feature_dir": str(subset_processed_root / "clip_embeddings"),
            "artifacts_dir": str(subset_artifacts_root),
            "stage1_projector_state": str(subset_artifacts_root / "projector_stage1.pkl"),
            "stage2_projector_state": str(subset_artifacts_root / "projector_stage2.pkl"),
            "stage2_llama_state": str(subset_artifacts_root / "llama_stage2.pkl"),
        }
    )

    out_config = (PROJECT_ROOT / args.output_config).resolve()
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(generated_cfg, indent=2), encoding="utf-8")

    manifest = {
        "profile_name": args.profile_name,
        "rows_requested": args.rows,
        "rows_selected": k,
        "seed": args.seed,
        "source_alignment_json": str(full_alignment_path),
        "source_chat_json": str(full_chat_path),
        "output_config": str(out_config),
    }
    manifest_path = subset_processed_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Created subset profile.")
    print("  rows_selected:", k)
    print("  profile_name:", args.profile_name)
    print("  output_config:", out_config)
    print("  subset_manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
