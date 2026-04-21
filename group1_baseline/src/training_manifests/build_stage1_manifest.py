"""Build stage-1 manifest for training loader.

Each manifest row contains:
- vision_path: path to precomputed CLIP .npy file
- input_ids: tokenized text ids
- labels: target labels for loss computation
"""

def build_stage1_manifest(tokenized_json, feature_dir, output_json, overwrite=False):
    """Create stage-1 manifest by pairing tokenized rows with CLIP feature files."""
    import json
    import os

    if os.path.exists(output_json) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_json}. Delete it first or set overwrite=True."
        )

    with open(tokenized_json, "r") as f:
        data = json.load(f)

    manifest = []
    for sample in data:
        # Use image filename to locate its precomputed CLIP embedding.
        manifest.append({
            "vision_path": os.path.join(
                feature_dir,
                sample["image"].replace(".jpg", ".npy")
            ),
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
        })

    # Training loop reads this manifest to batch data quickly.
    with open(output_json, "w") as f:
        json.dump(manifest, f)
