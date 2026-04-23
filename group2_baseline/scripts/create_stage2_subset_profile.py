"""Create deterministic 10k subset datasets for Group2 stage2 variants.

This script trims each variant's `stage2_dataset.jsonl` to a fixed row budget and
stores a manifest for reproducibility. It can sample in-place (source == target)
and will preserve the original file as `stage2_dataset_full.jsonl` once.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expand_project_root(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    token = "${PROJECT_ROOT}"
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        elif isinstance(v, list):
            out[k] = [x.replace(token, str(project_root)) if isinstance(x, str) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _expand_project_root(v, project_root)
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create deterministic Group2 JSONL subsets.")
    p.add_argument("--config", default="configs/workflow_paths_subset_10000.json")
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--source-root",
        default="",
        help="Optional source stage2 root. Defaults to config['stage2_source_root'] if present, else stage2_root.",
    )
    p.add_argument(
        "--manifest-name",
        default="subset_manifest.json",
        help="Manifest filename written under stage2_root.",
    )
    p.add_argument(
        "--allow-fallback-source",
        action="store_true",
        help="Allow fallback to existing stage2_root datasets when source_root files are missing.",
    )
    return p.parse_args()


def _read_jsonl_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _select_rows_by_image_ids(lines: list[str], keep_ids: set[int]) -> list[str]:
    selected: list[str] = []
    seen: set[int] = set()
    for line in lines:
        row = json.loads(line)
        image_id = int(row["image_id"])
        if image_id in keep_ids and image_id not in seen:
            selected.append(line)
            seen.add(image_id)
    return selected


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(cfg_raw, PROJECT_ROOT)

    stage2_root = Path(cfg["stage2_root"])
    source_root = Path(args.source_root) if args.source_root else Path(cfg.get("stage2_source_root", cfg["stage2_root"]))
    baseline_variant = str(cfg["baseline_variant"])
    quality_variants = [str(v) for v in cfg.get("quality_variants", [])]
    all_variants = [baseline_variant] + quality_variants

    rng = random.Random(args.seed)
    entries: list[dict[str, Any]] = []

    source_lines: dict[str, list[str]] = {}
    source_ids: dict[str, set[int]] = {}
    source_paths: dict[str, Path] = {}
    target_paths: dict[str, Path] = {}

    for variant in all_variants:
        src = source_root / variant / "stage2_dataset.jsonl"
        dst = stage2_root / variant / "stage2_dataset.jsonl"
        if not src.exists():
            if args.allow_fallback_source:
                fallback = stage2_root / variant / "stage2_dataset.jsonl"
                if fallback.exists():
                    print(
                        f"[create_stage2_subset_profile] source missing for '{variant}', "
                        f"falling back to existing stage2_root dataset: {fallback}"
                    )
                    src = fallback
                else:
                    raise FileNotFoundError(f"Missing source dataset for variant '{variant}': {src}")
            else:
                raise FileNotFoundError(
                    f"Missing source dataset for variant '{variant}': {src}. "
                    "Provide full source JSONLs under source_root or pass --allow-fallback-source intentionally."
                )
        lines = _read_jsonl_lines(src)
        ids = {int(json.loads(line)["image_id"]) for line in lines}
        source_lines[variant] = lines
        source_ids[variant] = ids
        source_paths[variant] = src
        target_paths[variant] = dst

    common_image_ids = sorted(set.intersection(*(source_ids[v] for v in all_variants)))
    k = min(args.rows, len(common_image_ids))
    if k == 0:
        raise RuntimeError("No common image_ids found across Group2 variants; cannot build subset profile.")
    selected_image_ids = set(rng.sample(common_image_ids, k))

    for variant in all_variants:
        src = source_paths[variant]
        dst = target_paths[variant]
        rows = source_lines[variant]
        subset = _select_rows_by_image_ids(rows, selected_image_ids)
        n = len(rows)
        selected_rows = len(subset)
        mode = "sampled_common_ids" if n > selected_rows else "copied_all"

        if dst.exists() and not args.overwrite and src != dst:
            write_mode = "skipped_existing"
        else:
            # If sampling in place, preserve a single full backup before overwrite.
            if src == dst and n > selected_rows:
                backup = dst.parent / "stage2_dataset_full.jsonl"
                if not backup.exists():
                    shutil.copy2(dst, backup)
            _write_jsonl_lines(dst, subset)
            write_mode = "generated"

        entries.append(
            {
                "variant": variant,
                "source_jsonl": str(src),
                "target_jsonl": str(dst),
                "source_rows": n,
                "selected_rows": selected_rows,
                "mode": mode,
                "write_mode": write_mode,
            }
        )

    manifest = {
        "rows_requested": args.rows,
        "selected_common_image_count": k,
        "common_image_count_before_sampling": len(common_image_ids),
        "seed": args.seed,
        "stage2_root": str(stage2_root),
        "source_root": str(source_root),
        "variants": entries,
    }
    manifest_path = stage2_root / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Created Group2 subset profile.")
    print("  stage2_root:", stage2_root)
    print("  source_root:", source_root)
    print("  rows_requested:", args.rows)
    print("  manifest:", manifest_path)
    for e in entries:
        print(
            f"  - {e['variant']}: source_rows={e['source_rows']} selected_rows={e['selected_rows']} "
            f"mode={e['mode']} write_mode={e['write_mode']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
