"""Run Stage 4.5 -> Stage 5 -> Stage 6 smoke flow on accelerator hosts.

Designed for environments where notebook UI is unreliable.

Usage:
  python scripts/run_tpu_smoke.py
  python scripts/run_tpu_smoke.py --max-rows 128 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_dotenv_file, load_json_config
from src.model_internals.loader_pipeline import ensure_llama_artifacts, load_llama_model_and_tokenizer
from src.training.train_pipeline import (
    build_smoke_manifest_from_existing_features,
    run_stage1_training_pipeline,
    run_stage2_training_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Group1 baseline smoke pipeline on TPU host.")
    parser.add_argument("--max-rows", type=int, default=256, help="Rows for each smoke manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite stage outputs if they exist.")
    parser.add_argument(
        "--use-mesh",
        action="store_true",
        default=True,
        help="Use JAX mesh during model load (default: true).",
    )
    parser.add_argument(
        "--no-mesh",
        dest="use_mesh",
        action="store_false",
        help="Disable JAX mesh during model load.",
    )
    parser.add_argument("--stage1-batch-size", type=int, default=1, help="Stage 1 smoke batch size.")
    parser.add_argument("--stage2-batch-size", type=int, default=1, help="Stage 2 smoke batch size.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Model dtype.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = PROJECT_ROOT
    load_dotenv_file(project_root / ".env")
    cfg = load_json_config(project_root / "configs" / "workflow_paths.json", project_root)
    try:
        import jax

        print("JAX backend:", jax.default_backend())
        print("JAX devices:", [str(d) for d in jax.devices()])
    except Exception as exc:
        print("WARNING: could not query JAX backend:", exc)

    stage1_smoke = Path(
        cfg.get(
            "stage1_manifest_smoke_json",
            str(project_root / "data" / "processed" / "stage1_alignment" / "stage1_manifest_smoke.json"),
        )
    )
    stage2_smoke = Path(
        cfg.get(
            "stage2_manifest_smoke_json",
            str(project_root / "data" / "processed" / "stage2_finetuning" / "stage2_manifest_smoke.json"),
        )
    )

    print("[1/5] Building smoke manifests...")
    s1_smoke = build_smoke_manifest_from_existing_features(
        source_manifest_json=Path(cfg["stage1_manifest_json"]),
        output_manifest_json=stage1_smoke,
        max_rows=args.max_rows,
    )
    s2_smoke = build_smoke_manifest_from_existing_features(
        source_manifest_json=Path(cfg["stage2_manifest_json"]),
        output_manifest_json=stage2_smoke,
        max_rows=args.max_rows,
    )
    print("  Stage1 smoke:", s1_smoke)
    print("  Stage2 smoke:", s2_smoke)

    print("[2/5] Ensuring LLaMA artifacts...")
    llama_dir = Path(
        cfg.get("llama_local_dir", str(project_root / "data" / "models" / "Llama-3.2-1B-Instruct"))
    )
    art = ensure_llama_artifacts(
        repo_id=cfg.get("tokenizer_id", "meta-llama/Llama-3.2-1B-Instruct"),
        local_dir=llama_dir,
    )
    print("  Artifacts:", art)

    print("[3/5] Loading model/tokenizer...")
    loaded = load_llama_model_and_tokenizer(local_dir=llama_dir, dtype=args.dtype, use_mesh=args.use_mesh)
    llama_model = loaded["llama_model"]
    print("  Model load:", {k: loaded[k] for k in ("llama_dir", "num_devices", "dtype", "mesh_enabled")})

    print("[4/5] Running Stage 5 smoke...")
    stage5 = run_stage1_training_pipeline(
        manifest_json=stage1_smoke,
        stage1_projector_state_path=Path(cfg["stage1_projector_state"]),
        llama_model=llama_model,
        learning_rate=1e-4,
        num_epochs=1,
        batch_size=args.stage1_batch_size,
        log_every=10,
        overwrite=args.overwrite,
        seed=0,
    )
    print("  Stage5:", stage5)

    print("[5/5] Running Stage 6 smoke...")
    stage6 = run_stage2_training_pipeline(
        manifest_json=stage2_smoke,
        stage1_projector_state_path=Path(cfg["stage1_projector_state"]),
        stage2_projector_state_path=Path(cfg["stage2_projector_state"]),
        stage2_llama_state_path=Path(cfg["stage2_llama_state"]),
        llama_model=llama_model,
        learning_rate=1e-5,
        num_epochs=1,
        batch_size=args.stage2_batch_size,
        log_every=10,
        overwrite=args.overwrite,
    )
    print("  Stage6:", stage6)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
