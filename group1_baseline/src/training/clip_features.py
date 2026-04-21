"""JIT CLIP feature extraction helpers from Group 1 notebook."""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def make_clip_feature_fn(clip_bundle):
    """Return a JIT-compiled function that extracts penultimate CLIP hidden state."""
    params = clip_bundle.model.params

    @jax.jit
    def get_features(pixel_values):
        outputs = clip_bundle.model(
            pixel_values=pixel_values,
            params=params,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-2]

    return get_features


def precompute_clip_features_jitted(clip_bundle, tokenized_json, image_root, output_dir, overwrite=False):
    """Precompute CLIP features exactly like notebook cell logic."""
    with open(tokenized_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    if not overwrite and any(name.endswith(".npy") for name in os.listdir(output_dir)):
        raise FileExistsError(
            f"Output dir already has .npy files: {output_dir}. "
            "Delete existing features first or set overwrite=True."
        )

    get_features_compiled = make_clip_feature_fn(clip_bundle)

    # Dataset rows can repeat the same image many times. Precompute each image once.
    unique_images = []
    seen = set()
    for sample in data:
        image_name = sample["image"]
        if image_name not in seen:
            seen.add(image_name)
            unique_images.append(image_name)

    print(f"Extracting features for {len(unique_images)} unique images...")
    for i, image_name in enumerate(unique_images):
        try:
            image_path = os.path.join(image_root, image_name)
            img = Image.open(image_path).convert("RGB")
            clip_inputs = clip_bundle.processor(images=img, return_tensors="np")
            pixel_values = jnp.array(clip_inputs["pixel_values"])
            hidden_states_penultimate = get_features_compiled(pixel_values)
            vision_feats = np.array(hidden_states_penultimate[0])

            save_path = os.path.join(output_dir, image_name.replace(".jpg", ".npy"))
            if os.path.exists(save_path) and not overwrite:
                # In case of partial rerun with overwrite=False, leave existing file untouched.
                continue
            np.save(save_path, vision_feats)

            if i % 100 == 0:
                print(f"Processed {i}/{len(unique_images)}...")
        except Exception as e:
            print(f"Error on {image_name}: {e}")
