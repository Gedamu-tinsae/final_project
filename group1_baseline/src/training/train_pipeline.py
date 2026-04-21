"""Training orchestration helpers for notebook-thin stage cells."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import optax
from flax import nnx

from .projector import VisionProjector
from .stage1 import run_stage1_training


def _infer_clip_dim_from_manifest(manifest_json: Path) -> int:
    with manifest_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"Manifest is empty: {manifest_json}")
    first_path = Path(manifest[0]["vision_path"])
    if not first_path.exists():
        raise FileNotFoundError(f"Missing vision feature file from manifest: {first_path}")
    feats = np.load(first_path)
    if feats.ndim != 2:
        raise ValueError(f"Unexpected feature shape {feats.shape} in {first_path}; expected [N_vis, D_clip].")
    return int(feats.shape[-1])


def run_stage1_training_pipeline(
    *,
    manifest_json: Path,
    stage1_projector_state_path: Path,
    llama_model,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    batch_size: int = 2,
    log_every: int = 10,
    overwrite: bool = False,
    seed: int = 0,
) -> dict:
    """Run stage-1 training and save projector state.

    Returns summary dict with mode and output path.
    """
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_json}")

    if stage1_projector_state_path.exists() and not overwrite:
        return {
            "mode": "skipped_existing",
            "projector_state_path": str(stage1_projector_state_path),
        }

    in_dim = _infer_clip_dim_from_manifest(manifest_json)
    out_dim = int(llama_model.config.embed_dim)

    projector = VisionProjector(in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(seed))
    projector_graphdef, projector_state = nnx.split(projector)
    llama_graphdef, llama_state = nnx.split(llama_model)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.0)
    # Flax NNX GraphState is a valid parameter tree at runtime, but Optax's
    # static type expects `Params`; cast keeps Pylance quiet without changing behavior.
    opt_state = tx.init(cast(Any, projector_state))

    projector_state, _ = run_stage1_training(
        manifest_json=str(manifest_json),
        projector_state=projector_state,
        opt_state=opt_state,
        projector_graphdef=projector_graphdef,
        llama_state=llama_state,
        llama_graphdef=llama_graphdef,
        tx=tx,
        num_epochs=num_epochs,
        batch_size=batch_size,
        log_every=log_every,
    )

    stage1_projector_state_path.parent.mkdir(parents=True, exist_ok=True)
    with stage1_projector_state_path.open("wb") as f:
        pickle.dump(projector_state, f)

    return {
        "mode": "generated",
        "projector_state_path": str(stage1_projector_state_path),
        "in_dim": in_dim,
        "out_dim": out_dim,
    }
