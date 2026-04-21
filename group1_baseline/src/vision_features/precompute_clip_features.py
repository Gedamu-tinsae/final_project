"""Precompute CLIP vision embeddings and save them to disk.

Why this exists:
- Running CLIP every training step is expensive.
- This script computes image features once and stores `.npy` files.
- Training then loads the precomputed features directly.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from PIL import Image


def precompute_clip_features(
    clip_bundle,
    tokenized_json,
    image_root,
    output_dir,
):
    """Create one .npy CLIP feature file per sample image in tokenized_json."""
    with open(tokenized_json, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(data):
        # Resolve image path from root + filename in dataset entry.
        image_path = os.path.join(image_root, sample["image"])
        img = Image.open(image_path).convert("RGB")

        # Process image and run CLIP vision tower.
        clip_inputs = clip_bundle.processor(images=img, return_tensors="np")
        pixel_values = jnp.array(clip_inputs["pixel_values"])

        vision_outputs = clip_bundle.model.vision_model(pixel_values=pixel_values)
        vision_feats = np.array(vision_outputs.last_hidden_state[0])  # [N_vis, D_clip]

        # Save features with same basename as image.
        save_path = os.path.join(output_dir, sample["image"].replace(".jpg", ".npy"))
        np.save(save_path, vision_feats)
