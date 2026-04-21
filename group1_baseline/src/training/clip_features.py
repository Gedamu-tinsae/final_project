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


def precompute_clip_features_jitted(clip_bundle, tokenized_json, image_root, output_dir):
    """Precompute CLIP features exactly like notebook cell logic."""
    with open(tokenized_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    get_features_compiled = make_clip_feature_fn(clip_bundle)

    print(f"Extracting features for {len(data)} images...")
    for i, sample in enumerate(data):
        try:
            image_path = os.path.join(image_root, sample["image"])
            img = Image.open(image_path).convert("RGB")
            clip_inputs = clip_bundle.processor(images=img, return_tensors="np")
            pixel_values = jnp.array(clip_inputs["pixel_values"])
            hidden_states_penultimate = get_features_compiled(pixel_values)
            vision_feats = np.array(hidden_states_penultimate[0])

            save_path = os.path.join(output_dir, sample["image"].replace(".jpg", ".npy"))
            np.save(save_path, vision_feats)

            if i % 100 == 0:
                print(f"Processed {i}/{len(data)}...")
        except Exception as e:
            print(f"Error on {sample['image']}: {e}")
