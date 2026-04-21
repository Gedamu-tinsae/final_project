from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image

from .tokenization import resolve_stage2_paths


def _resolve_feature_path(image_name: str, feature_roots: list[Path]) -> Path | None:
    rel = Path(image_name).with_suffix(".npy")
    for root in feature_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def extract_stage2_features(
    stage2_root: Path,
    image_root: Path,
    feature_root: Path,
    clip_bundle,
    get_features_compiled,
    variant: str,
    data_split: str = "train",
    overwrite: bool = False,
    additional_feature_roots: list[Path] | None = None,
) -> dict:
    _, _, tokenized_json = resolve_stage2_paths(stage2_root, variant, data_split)
    stage2_data = json.loads(tokenized_json.read_text(encoding="utf-8"))
    feature_roots = [feature_root] + list(additional_feature_roots or [])

    created_features = 0
    skipped_features = 0
    reused_features = 0
    missing_images = 0
    failed_images = 0

    for row in stage2_data:
        image_name = row["image"]
        image_path = image_root / image_name
        feature_path = (feature_root / Path(image_name)).with_suffix(".npy")
        feature_path.parent.mkdir(parents=True, exist_ok=True)

        existing = _resolve_feature_path(image_name, feature_roots)
        if existing is not None and not overwrite:
            skipped_features += 1
            if existing != feature_path:
                reused_features += 1
            continue
        if not image_path.exists():
            missing_images += 1
            continue
        try:
            img = Image.open(image_path).convert("RGB")
            clip_inputs = clip_bundle.processor(images=img, return_tensors="np")
            pixel_values = jnp.asarray(clip_inputs["pixel_values"])
            hidden_states_penultimate = get_features_compiled(pixel_values)
            vision_feats = np.array(jnp.asarray(hidden_states_penultimate[0], dtype=jnp.float32))
            np.save(feature_path, vision_feats)
            created_features += 1
        except Exception:
            failed_images += 1

    return {
        "variant": variant,
        "split": data_split,
        "created_features": created_features,
        "skipped_features": skipped_features,
        "reused_features": reused_features,
        "missing_images": missing_images,
        "failed_images": failed_images,
    }
