"""Convert stage-1 alignment rows to instruction-response format.

Input rows:
  - image
  - caption

Output rows:
  - image
  - instruction (sampled from PROMPTS)
  - response (the original caption)
"""

import argparse
import json
import random
from pathlib import Path

# Prompt pool used to turn captioning into instruction-following format.
PROMPTS = [
    "Describe this image briefly.",
    "What is in this image?",
    "Provide a short description of the image.",
    "Summarize this picture."
]

def convert_alignment_rows(data, seed=42):
    """Convert alignment rows to instruction/response rows."""
    random.seed(seed)
    formatted = []
    for sample in data:
        formatted.append(
            {
                "image": sample["image"],
                "instruction": random.choice(PROMPTS),
                "response": sample["caption"],
            }
        )
    return formatted

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/stage1_alignment/alignment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stage1_alignment/alignment_chat.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Delete it first or run with --overwrite."
        )
    # Read stage-1 alignment rows from previous step.
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = convert_alignment_rows(data, seed=args.seed)

    # Save converted instruction-style dataset.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)
    print(f"Wrote {len(formatted)} rows to {args.output}")


if __name__ == "__main__":
    main()
