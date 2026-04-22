"""Validate Group4 prerequisites from Group1/Group2 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    p = argparse.ArgumentParser(description="Check Group4 prerequisites.")
    p.add_argument("--config", default="configs/workflow_paths.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = _expand_project_root(raw_cfg, PROJECT_ROOT)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CONFIG:", config_path)
    print()

    req = cfg["required_inputs"]
    missing: list[str] = []
    for key, value in req.items():
        p = Path(value)
        exists = p.exists()
        print(f"{'OK' if exists else 'MISSING'}: {key} -> {p}")
        if not exists:
            missing.append(key)

    print()
    if missing:
        print("Group4 is blocked by missing prerequisites:")
        for key in missing:
            print("-", key)
        return 2

    print("All Group4 prerequisites are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
