from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from .common import write_json


def build_heldout_eval_pack(
    stage2_root: Path,
    all_variants: list[str],
    samples_per_task: int = 10,
    seed: int = 123,
    overwrite: bool = False,
) -> dict:
    out_path = stage2_root / "heldout_eval_pack.json"
    if out_path.exists() and not overwrite:
        return {"mode": "skipped_existing", "path": str(out_path)}
    split_path = stage2_root / "shared_split.json"
    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    val_ids = set(split_info["val_image_ids"])

    def load_val_rows_by_key(variant: str) -> dict:
        rows_by_key = {}
        with (stage2_root / variant / "stage2_dataset.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["image_id"] not in val_ids:
                    continue
                key = (row["image_id"], row["task_type"])
                rows_by_key[key] = row
        return rows_by_key

    variant_maps = {variant: load_val_rows_by_key(variant) for variant in all_variants}
    common_keys = None
    for variant in all_variants:
        keys = set(variant_maps[variant].keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = sorted(common_keys or [])

    aligned_keys = []
    for key in common_keys:
        instructions = [variant_maps[v][key]["instruction"] for v in all_variants]
        if len(set(instructions)) == 1:
            aligned_keys.append(key)

    keys_by_task = defaultdict(list)
    for key in aligned_keys:
        keys_by_task[key[1]].append(key)

    rng = random.Random(seed)
    selected_keys = []
    for _, keys in sorted(keys_by_task.items()):
        pool = list(keys)
        rng.shuffle(pool)
        selected_keys.extend(pool[:samples_per_task])

    samples = []
    for key in selected_keys:
        image_id, task_type = key
        base_row = variant_maps[all_variants[0]][key]
        samples.append(
            {
                "image_id": image_id,
                "task_type": task_type,
                "instruction": base_row["instruction"],
                "reference_responses": {variant: variant_maps[variant][key]["response"] for variant in all_variants},
            }
        )

    payload = {"mode": "generated", "all_variants": all_variants, "num_samples": len(samples), "samples_per_task": samples_per_task, "samples": samples}
    write_json(out_path, payload, overwrite=True)
    return payload
