"""CLI orchestration for Group1 baseline workflow (notebook-equivalent core stages).

Stages covered:
1) Data acquisition + prep
2) Tokenization
3) CLIP feature precompute
4) Manifest build
5) Stage1 training (optional)
6) Stage2 training (optional)
7) Artifact existence check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_dotenv_file, load_json_config
from src.data_prep.stage1_pipeline import run_stage1_data_prep
from src.model_internals.loader_pipeline import ensure_llama_artifacts, load_llama_model_and_tokenizer
from src.training.tokenization_pipeline import run_tokenization_pipeline
from src.training.train_pipeline import (
    build_smoke_manifest_from_existing_features,
    run_stage1_training_pipeline,
    run_stage2_training_pipeline,
)
from src.training_manifests.manifest_pipeline import run_manifest_pipeline
from src.vision_features.feature_pipeline import run_stage1_clip_precompute


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Group1 baseline core workflow with safe guards.")
    p.add_argument("--config", default="configs/workflow_paths.json")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    p.add_argument("--download", action="store_true", help="Enable COCO download in Stage 1.")
    p.add_argument("--extract", action="store_true", help="Enable archive extraction in Stage 1.")
    p.add_argument(
        "--train",
        choices=["none", "stage1", "stage2", "both"],
        default="none",
        help="Which training stages to run.",
    )
    p.add_argument("--smoke", action="store_true", help="Use smoke manifests for training stages.")
    p.add_argument("--smoke-rows", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--no-mesh", action="store_true", help="Disable model mesh for training bootstrap.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv_file(dotenv_path)
    cfg = load_json_config(config_path, PROJECT_ROOT)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CONFIG:", config_path)
    print("HF_TOKEN loaded:", bool(os.environ.get("HF_TOKEN")))

    # Stage 1
    print("\n[Stage 1] Data acquisition + prep")
    alignment, chat_rows, mode, status = run_stage1_data_prep(
        project_root=PROJECT_ROOT,
        coco_json_path=Path(cfg["coco_json"]),
        alignment_path=Path(cfg["stage1_alignment_json"]),
        chat_path=Path(cfg["stage1_chat_json"]),
        seed=42,
        overwrite=args.overwrite,
        download=args.download,
        extract=args.extract,
    )
    print("mode:", mode, "alignment_rows:", len(alignment), "chat_rows:", len(chat_rows))
    for k, pth in status.items():
        print(f"  {k}: {'OK' if pth.exists() else 'MISSING'} -> {pth}")

    # Stage 2
    print("\n[Stage 2] Tokenization")
    tok = run_tokenization_pipeline(
        tokenizer_id=cfg.get("tokenizer_id", "meta-llama/Llama-3.2-1B-Instruct"),
        stage1_input_json=Path(cfg["stage1_chat_json"]),
        stage1_output_json=Path(cfg["stage1_tokenized_json"]),
        stage2_input_json=Path(cfg["stage2_input_json"]),
        stage2_output_json=Path(cfg["stage2_tokenized_json"]),
        stage1_max_len=128,
        stage2_max_len=256,
        overwrite=args.overwrite,
    )
    print("stage1:", tok["stage1_mode"], "stage2:", tok["stage2_mode"])

    # Stage 3
    print("\n[Stage 3] CLIP feature precompute")
    feat = run_stage1_clip_precompute(
        tokenized_json=Path(cfg["stage1_tokenized_json"]),
        image_root=Path(cfg["image_root"]),
        output_dir=Path(cfg["clip_feature_dir"]),
        clip_model_dir=cfg.get("clip_model_dir"),
        download_if_missing=True,
        overwrite=args.overwrite,
    )
    print("mode:", feat["mode"], "feature_files:", feat["num_feature_files"])

    # Stage 4
    print("\n[Stage 4] Manifest build")
    man = run_manifest_pipeline(
        stage1_tokenized_json=Path(cfg["stage1_tokenized_json"]),
        stage2_tokenized_json=Path(cfg["stage2_tokenized_json"]),
        clip_feature_dir=Path(cfg["clip_feature_dir"]),
        stage1_manifest_json=Path(cfg["stage1_manifest_json"]),
        stage2_manifest_json=Path(cfg["stage2_manifest_json"]),
        overwrite=args.overwrite,
    )
    print("stage1:", man["stage1_mode"], "rows:", man["stage1_rows"])
    print("stage2:", man["stage2_mode"], "rows:", man["stage2_rows"])

    run_stage1 = args.train in ("stage1", "both")
    run_stage2 = args.train in ("stage2", "both")
    llama_model = None
    if run_stage1 or run_stage2:
        print("\n[Stage 4.5] Model bootstrap")
        llama_dir = Path(cfg.get("llama_local_dir", str(PROJECT_ROOT / "data" / "models" / "Llama-3.2-1B-Instruct")))
        art = ensure_llama_artifacts(
            repo_id=cfg.get("tokenizer_id", "meta-llama/Llama-3.2-1B-Instruct"),
            local_dir=llama_dir,
        )
        print("artifacts:", art["mode"], "->", art["local_dir"])
        loaded = load_llama_model_and_tokenizer(local_dir=llama_dir, dtype=args.dtype, use_mesh=not args.no_mesh)
        llama_model = loaded["llama_model"]
        print("model:", {k: loaded[k] for k in ("num_devices", "dtype", "mesh_enabled")})

    if run_stage1 and llama_model is not None:
        print("\n[Stage 5] Stage1 training")
        stage1_manifest = Path(cfg["stage1_manifest_json"])
        if args.smoke:
            stage1_manifest = Path(
                cfg.get(
                    "stage1_manifest_smoke_json",
                    str(PROJECT_ROOT / "data" / "processed" / "stage1_alignment" / "stage1_manifest_smoke.json"),
                )
            )
            build_smoke_manifest_from_existing_features(
                source_manifest_json=Path(cfg["stage1_manifest_json"]),
                output_manifest_json=stage1_manifest,
                max_rows=args.smoke_rows,
            )
        out = run_stage1_training_pipeline(
            manifest_json=stage1_manifest,
            stage1_projector_state_path=Path(cfg["stage1_projector_state"]),
            llama_model=llama_model,
            learning_rate=1e-4,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=10,
            overwrite=args.overwrite,
            seed=0,
        )
        print("mode:", out["mode"], "projector:", out["projector_state_path"])

    if run_stage2 and llama_model is not None:
        print("\n[Stage 6] Stage2 training")
        stage2_manifest = Path(cfg["stage2_manifest_json"])
        if args.smoke:
            stage2_manifest = Path(
                cfg.get(
                    "stage2_manifest_smoke_json",
                    str(PROJECT_ROOT / "data" / "processed" / "stage2_finetuning" / "stage2_manifest_smoke.json"),
                )
            )
            build_smoke_manifest_from_existing_features(
                source_manifest_json=Path(cfg["stage2_manifest_json"]),
                output_manifest_json=stage2_manifest,
                max_rows=args.smoke_rows,
            )
        out = run_stage2_training_pipeline(
            manifest_json=stage2_manifest,
            stage1_projector_state_path=Path(cfg["stage1_projector_state"]),
            stage2_projector_state_path=Path(cfg["stage2_projector_state"]),
            stage2_llama_state_path=Path(cfg["stage2_llama_state"]),
            llama_model=llama_model,
            learning_rate=1e-5,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=10,
            overwrite=args.overwrite,
        )
        print("mode:", out["mode"])
        print("stage2 projector:", out["stage2_projector_state_path"])
        print("stage2 llama:", out["stage2_llama_state_path"])

    print("\n[Stage 7] Artifact check")
    checks = [
        cfg["stage1_chat_json"],
        cfg["stage1_tokenized_json"],
        cfg["stage1_manifest_json"],
        cfg["stage2_tokenized_json"],
        cfg["stage2_manifest_json"],
    ]
    if run_stage1:
        checks.append(cfg["stage1_projector_state"])
    if run_stage2:
        checks.extend([cfg["stage2_projector_state"], cfg["stage2_llama_state"]])
    for p in checks:
        path = Path(p)
        print(f"{'OK' if path.exists() else 'MISSING'} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

