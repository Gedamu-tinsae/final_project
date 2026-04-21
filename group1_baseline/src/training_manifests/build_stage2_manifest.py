"""Build stage-2 manifest for multimodal finetuning."""

import json
import os

def build_stage2_manifest(tokenized_json, feature_dir, output_json, overwrite=False):
    """Create stage-2 manifest with vision feature paths + tokenized text."""
    if os.path.exists(output_json) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_json}. Delete it first or set overwrite=True."
        )

    with open(tokenized_json, "r") as f:
        data = json.load(f)

    manifest = []
    for sample in data:
        # Stage-2 still reuses CLIP feature files keyed by image name.
        manifest.append({
            "vision_path": os.path.join(
                feature_dir,
                sample["image"].replace(".jpg", ".npy"),
            ),
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
        })

    # Output consumed by stage-2 training loop.
    with open(output_json, "w") as f:
        json.dump(manifest, f)
