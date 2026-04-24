"""Thin CLI wrapper for Group4 PEFT smoke pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for p in (PROJECT_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.group4_pipeline.peft_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
