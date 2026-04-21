"""Orchestration helpers for CLIP feature precompute."""

from __future__ import annotations

from pathlib import Path

from .clip_helpers import create_clip_from_flax_checkpoint
from ..training.clip_features import precompute_clip_features_jitted


def run_stage1_clip_precompute(
    *,
    tokenized_json: Path,
    image_root: Path,
    output_dir: Path,
    clip_model_dir: str | None = None,
    download_if_missing: bool = True,
    overwrite: bool = False,
) -> dict:
    """Run CLIP precompute for Stage 1 tokenized dataset."""
    if not tokenized_json.exists():
        raise FileNotFoundError(f"Missing tokenized input: {tokenized_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_dir.glob("*.npy")))
    if existing > 0 and not overwrite:
        return {
            "mode": "skipped_existing",
            "output_dir": str(output_dir),
            "num_feature_files": existing,
        }

    kwargs = {"download_if_missing": download_if_missing}
    if clip_model_dir:
        kwargs["local_dir"] = clip_model_dir
    clip_bundle = create_clip_from_flax_checkpoint(**kwargs)

    precompute_clip_features_jitted(
        clip_bundle=clip_bundle,
        tokenized_json=str(tokenized_json),
        image_root=str(image_root),
        output_dir=str(output_dir),
        overwrite=overwrite,
    )

    num_features = len(list(output_dir.glob("*.npy")))
    return {"mode": "generated", "output_dir": str(output_dir), "num_feature_files": num_features}
