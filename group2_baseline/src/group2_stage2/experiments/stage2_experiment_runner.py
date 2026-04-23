from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Callable, cast

import optax
from flax import nnx

try:
    from group1_baseline.src.model_internals.loader_pipeline import load_llama_model_and_tokenizer
    from group1_baseline.src.training.batching import pad_list
    from group1_baseline.src.training.projector import VisionProjector
    from group1_baseline.src.training.stage2 import eval_step_stage2, train_step_stage2
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from group1_baseline.src.model_internals.loader_pipeline import load_llama_model_and_tokenizer
    from group1_baseline.src.training.batching import pad_list
    from group1_baseline.src.training.projector import VisionProjector
    from group1_baseline.src.training.stage2 import eval_step_stage2, train_step_stage2

from .training_orchestration import run_stage2_training


def _infer_clip_dim_from_manifest(manifest_json: Path) -> int:
    import json
    import numpy as np

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError(f"Manifest is empty: {manifest_json}")
    vision_path = Path(manifest[0]["vision_path"])
    if not vision_path.exists():
        raise FileNotFoundError(f"Missing vision feature file from manifest: {vision_path}")
    feats = np.load(vision_path, allow_pickle=False)
    if feats.ndim != 2:
        raise ValueError(f"Unexpected vision feature shape {feats.shape} in {vision_path}; expected [N_vis, D_clip].")
    return int(feats.shape[-1])


def _resolve_group1_defaults(project_root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    profile = str(cfg.get("subset_profile_name", "subset_10000_seed42"))
    group1_root = (project_root / "../group1_baseline").resolve()
    llama_local_dir = Path(
        cfg.get("llama_local_dir", str(group1_root / "data/models/Llama-3.2-1B-Instruct"))
    ).resolve()
    if "stage1_projector_state" in cfg:
        stage1_projector_state_path = Path(str(cfg["stage1_projector_state"])).resolve()
    else:
        subset_default = (group1_root / f"artifacts/subsets/{profile}/projector_stage1.pkl").resolve()
        legacy_default = (group1_root / "artifacts/projector_stage1.pkl").resolve()
        stage1_projector_state_path = subset_default if subset_default.exists() else legacy_default
    return {
        "group1_root": group1_root,
        "llama_local_dir": llama_local_dir,
        "stage1_projector_state_path": stage1_projector_state_path,
    }


def run_stage2_experiment(
    *,
    project_root: Path,
    stage2_root: Path,
    variant: str,
    cfg: dict[str, Any],
    num_epochs: int = 1,
    batch_size: int = 8,
    log_every_steps: int = 20,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    dtype: str = "bfloat16",
    use_mesh: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one Group2 variant Stage-2 experiment from a clean initialization."""
    variant_root = stage2_root / variant
    train_manifest = variant_root / "stage2_manifest_train.json"
    val_manifest = variant_root / "stage2_manifest_val.json"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Missing val manifest: {val_manifest}")

    refs = _resolve_group1_defaults(project_root, cfg)
    stage1_projector_state_path = refs["stage1_projector_state_path"]
    llama_local_dir = refs["llama_local_dir"]
    if not stage1_projector_state_path.exists():
        raise FileNotFoundError(f"Missing Stage1 projector state: {stage1_projector_state_path}")
    if not llama_local_dir.exists():
        raise FileNotFoundError(f"Missing LLaMA local dir: {llama_local_dir}")

    loaded = load_llama_model_and_tokenizer(
        local_dir=llama_local_dir,
        dtype=dtype,
        use_mesh=use_mesh,
    )
    llama_model = loaded["llama_model"]

    with stage1_projector_state_path.open("rb") as f:
        projector_state = pickle.load(f)

    in_dim = _infer_clip_dim_from_manifest(train_manifest)
    out_dim = int(llama_model.config.embed_dim)
    projector = VisionProjector(in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(seed))
    projector_graphdef, _ = nnx.split(projector)
    llama_graphdef, llama_state = nnx.split(llama_model)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = tx.init(cast(Any, {"projector": projector_state, "llama": llama_state}))

    def _make_train_step() -> Callable[[dict], float]:
        nonlocal projector_state, llama_state, opt_state

        def _fn(batch: dict) -> float:
            nonlocal projector_state, llama_state, opt_state
            projector_state, llama_state, opt_state, loss = train_step_stage2(
                projector_state,
                llama_state,
                opt_state,
                batch,
                projector_graphdef,
                llama_graphdef,
                tx,
            )
            return float(loss)

        return _fn

    def _make_eval_step() -> Callable[[dict], float]:
        def _fn(batch: dict) -> float:
            return float(eval_step_stage2(projector_state, llama_state, batch, projector_graphdef, llama_graphdef))

        return _fn

    training_result = run_stage2_training(
        manifest_json=train_manifest,
        val_manifest_json=val_manifest,
        batch_size=batch_size,
        num_epochs=num_epochs,
        log_every_steps=log_every_steps,
        train_step_fn=_make_train_step(),
        eval_step_fn=_make_eval_step(),
        pad_list=pad_list,
    )

    return {
        "variant": variant,
        "train_result": training_result,
        "val_result": training_result["final_val_result"],
        "final_train_mean_loss": training_result["final_train_mean_loss"],
        "history": training_result["history"],
        "manifests": {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
        },
        "runtime": {
            "llama_local_dir": str(llama_local_dir),
            "stage1_projector_state_path": str(stage1_projector_state_path),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "log_every_steps": int(log_every_steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "dtype": dtype,
            "use_mesh": bool(use_mesh),
        },
    }
