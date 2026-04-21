"""Stage-1 data prep from COCO captions.

This script converts COCO's annotations format into a simple list where each
row contains:
  - image: image filename
  - caption: one caption for that image
"""

import argparse
import json
from pathlib import Path


def build_alignment(coco_json_path: Path) -> list[dict]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build quick lookup: image_id -> filename.
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    dataset = []

    # Expand each caption annotation into a simple training row.
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id in id_to_filename:
            dataset.append(
                {
                    "image": id_to_filename[image_id],
                    "caption": ann["caption"],
                }
            )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-json",
        type=Path,
        default=Path("data/raw/annotations/captions_train2017.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stage1_alignment/alignment.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_alignment(args.coco_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
