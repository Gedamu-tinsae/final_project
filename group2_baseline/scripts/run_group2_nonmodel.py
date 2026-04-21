from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.group2_stage2.audit import audit_stage2_variants
from src.group2_stage2.splits import build_shared_quality_pool, materialize_train_val_split


def _expand_project_root(cfg: dict, project_root: Path) -> dict:
    token = "${PROJECT_ROOT}"
    out: dict = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            out[k] = v.replace(token, str(project_root))
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Group2 non-model prep (audit + shared split).")
    parser.add_argument("--config", default="configs/workflow_paths.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(raw_cfg, project_root)
    stage2_root = Path(cfg["stage2_root"])
    baseline_variant = cfg["baseline_variant"]
    quality_variants = cfg.get("quality_variants", [])
    all_variants = [baseline_variant] + quality_variants

    audit = audit_stage2_variants(stage2_root, all_variants)
    print("audit:", json.dumps(audit, indent=2))

    pool = build_shared_quality_pool(
        stage2_root=stage2_root,
        all_variants=all_variants,
        quality_image_count=int(cfg["quality_image_count"]),
        val_image_count=int(cfg["val_image_count"]),
        split_seed=int(cfg.get("split_seed", 42)),
        pool_reference_variant=baseline_variant,
        overwrite=args.overwrite,
    )
    print("shared pool:", json.dumps(pool, indent=2))

    split = materialize_train_val_split(stage2_root, all_variants, overwrite=args.overwrite)
    print("split materialization:", json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
